Meal Planner (JSON → 4-Week Plan)

A lightweight CLI that generates a personalized 4-week meal plan from JSON diet templates.
It computes calorie needs (BMR/TDEE), selects the right calorie band, rewrites options to respect dietary rules/dislikes, and assembles a schedule with controlled repetition so it feels natural (not too samey, not too random).

Why this exists (the core idea)

Start from authored options (JSON templates per calorie band).

Personalize them:

Compute target calories from user metrics and goal.

Filter/transform options based on dietary rules (vegan/vegetarian, dairy-free, gluten-free) and dislikes (seafood, eggs, spicy foods, etc.).

Schedule meals for 4 weeks with probabilistic repetition control:

Within the same day, if two adjacent slots share the same option pool, avoid picking the exact same option 60% of the time (allowing repeats 40%).

Across days, avoid reusing yesterday’s option for the same slot 60% of the time.

The result: realistic variety without being rigid or repetitive.

What this project uses

Language: Python 3.8+ (standard library only)

CLI: input() prompts (no external UI)

Data: JSON diet templates per calorie band (e.g., 2400_2600_kcal.json)

Outputs:

Console: formatted 4-week plan

CSV: 4_Week_Meal_Plan.csv (tabular schedule)

CSV: user_info.csv (run inputs & computed metrics)

No third-party packages required.

How it works (architecture)
1) Intake & calculations

Prompts: gender, height (cm), weight (kg), age, goal (lose/gain), activity level, meals/day, allergies/dislikes.

BMR (Mifflin–St Jeor) → TDEE (activity factor).

Applies goal adjustment: −500 kcal for loss, +300 kcal for gain.

Maps to a calorie band and picks the corresponding JSON file (e.g., 2532 kcal → 2400_2600_kcal.json).

2) JSON diet templates

Two supported shapes (both end up in one internal format):

Internal format: direct slot_sequence + slots with options.

Structures format: templated meals_per_day + meals entries; auto-converted.

The planner preserves authored slot order (e.g., Meal 1 → Meal 2 → Snack → Meal 3 → Meal 4).

3) Preference engine (rewriting & filtering)

High-level rules: Vegan, Vegetarian, Dairy-free, Gluten-free.

Dislike groups: seafood, red meat, poultry, pork, eggs, legumes, soy, nuts & seeds, cruciferous, nightshades, starchy veg, sweets, spicy foods.

Text-aware rewriting:

Detects tokens in option text (e.g., “chicken breast”, “eggs”, “yogurt”, “pasta”).

Swaps to allowed replacements (e.g., tofu, fava beans, coconut yogurt, rice, etc.).

Cleans artifacts and de-duplicates “+” items.

Final pass filters out options that still violate constraints (fallback keeps originals if all would be removed).

4) Controlled repetition (the 60/40 logic)

Within a day: when two adjacent slots share the same option set, the second slot avoids the previous pick 60% of the time (so repeats still happen ~40%).

Across days: for each slot label (e.g., “meal 3”), avoid reusing yesterday’s pick 60% of the time.

Uses a single RNG (seedable) so results can be reproducible.

5) Output

Key modules & responsibilities

BMR/TDEE & band selection

bmr, TDEE, target_kcal, assign_category, compute_band_filename_from_target.

Template loading & normalization

load_diets_from_band, _convert_json_structure_to_internal, _slot_key_from_json_meal_number.

Preference engine

Token detection: contains_any, KEYS.

Rewriting: rewrite_option_text_for_prefs, _token_replace, _cleanup_artifacts, _dedupe_plus_items.

Replacement pools: (e.g., DAIRY/EGG/GLUTEN replacements).

Enforcement: option_violates_prefs, filter_diet_options_by_prefs.

Diversity logic

_same_option_set: compares two slot option pools.

_pick_option_index: probabilistic avoidance (60% avoid).

build_one_day_from_diet: within-day + across-day avoidance.

build_plan_from_diets: tracks “yesterday’s pick” per slot.

CLI & output

run_planner: prompts, prints, writes 4_Week_Meal_Plan.csv and user_info.csv.

Extending the project

Add new replacements: extend the REPLACEMENT_POOLS lists (e.g., more dairy alternatives).

Add new dislike categories: update KEYS (token list) + DISLIKE_GROUP_OPTIONS + option_violates_prefs mapping.

Tune repetition: change the 60/40 probabilities in build_one_day_from_diet and _pick_option_index.

More slots: include snack 2, snack 3, or meal 5/6 in slot_sequence of templates.

Limitations / assumptions

Rewriting is text-based. It doesn’t re-calculate calories for replaced items (macros logic is scaffolded for future expansion).

Requires well-formed JSON with realistic options per slot; empty bands reduce variety.

The 60/40 logic controls selection, not recipe authoring — if two slots have tiny option pools, repeats can still happen (by design).

Pretty printed plan grouped by week → day → slots.

CSV rows with Week, DayInWeek, AbsoluteDay, Diet Label, Diet Kcal, Slot, Option Ref, Option Name.

Run metadata appended to user_info.csv.
