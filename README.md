Meal Planner (Excel + JSON → 4-Week Personalized Plan)

This project generates a personalized 4-week meal plan by combining two sources of information:

An authored diet template (JSON) → defines the structure of meals per day and suggested food options for different calorie bands.

A structured nutrition database (Excel) → contains foods, macronutrient information, and tags used for substitutions and dietary adjustments.

The system calculates calorie needs, selects the correct diet band, rewrites food options to respect dietary rules and dislikes, and assembles a realistic schedule with controlled repetition.
The goal is a plan that feels varied but natural.

Why this exists

Most diet planners are either rigid (fixed meal plans) or chaotic (random food suggestions).
This system finds the middle ground:

Starts from curated meal templates.

Personalizes them to the individual’s calorie needs and preferences.

Uses food substitutions to adapt recipes automatically.

Applies probability rules to balance variety vs. repetition.

The result is a 4-week plan that feels authored, while still flexible and personalized.

How it works
1. User Profile & Calculations

The system collects: gender, age, height, weight, activity level, diet goal (lose or gain), meals per day, and dietary preferences.

Calorie needs are calculated:

Basal Metabolic Rate (BMR) → Total Daily Energy Expenditure (TDEE).

Adjusted for the goal (−500 kcal/day for weight loss, +300 kcal/day for gain).

The result is mapped to a calorie band (e.g. 2400–2600 kcal).

2. JSON Diet Templates

Each calorie band has a JSON template.

A template defines:

Meal slots per day (Meal 1, Meal 2, Snack, etc.).

Options for each slot (different foods or meals to choose from).

Two formats are supported, both converted to the same internal structure:

Direct slots + options.

Structured meals_per_day + meals list.

3. Excel Nutrition Database

The Excel file contains foods, calories, protein, carbs, fat, and functional tags.

Foods are grouped by type (protein, carb, fat, etc.) and sub-category.

The database allows:

Recognizing foods mentioned in the JSON templates.

Finding substitutes when a food conflicts with dietary rules (e.g. swapping cow’s milk for oat milk).

Maintaining calories by adjusting portion sizes when swapping.

4. Preference Engine

The system rewrites meal options to respect rules and dislikes:

Diet rules: Vegan, Vegetarian, Dairy-free, Gluten-free.

Dislike groups: Seafood, Poultry, Pork, Red meat, Eggs, Legumes, Soy, Nuts & seeds, Starchy vegetables, Spicy foods, etc.

The text of each option is scanned for keywords.

Foods that violate preferences are swapped with alternatives from the Excel DB.

Portion sizes are recalculated to keep calories in line.

A final cleanup removes duplicates and odd artifacts in text.

5. Diversity Logic (60/40 Rule)

Meal variety is managed with probabilistic rules:

Within a day: If two adjacent meals share the same option pool, the second avoids repeating the same pick 60% of the time (but allows 40% repetition).

Across days: The same slot (e.g. “Meal 3”) avoids repeating yesterday’s option 60% of the time.

This ensures a balance of natural repetition and healthy variety.

6. Output

The system produces:

A 4-week schedule of meals, grouped by week and day.

Each day includes the selected options for every slot, rewritten to reflect preferences.

Summary files that contain both the plan and the user profile with calculations.

Key Modules in the System

Calorie Calculations → BMR, TDEE, goal adjustment, calorie band assignment.

Template Normalization → Load JSON diets and convert them to a uniform internal structure.

Excel Food Database → Load foods, macros, and tags for replacements.

Preference Engine → Rewrite meal options to respect rules/dislikes and maintain calories.

Diversity Logic → Apply 60/40 probability rules for realistic variety.

Plan Builder → Assemble 28 days of meals from the selected diet band.

Export → Save plan and user profile in structured formats.

System Flow (Conceptual)

Collect user information.

Compute calorie needs and assign a calorie band.

Load the matching JSON template.

Load the Excel nutrition database.

Rewrite meal options using preference rules and substitutions.

Apply repetition logic to schedule meals.

Build a 4-week plan with balanced variety.

Output the plan and user profile.

Limitations & Assumptions

Food substitutions are rule-based, not AI-generated recipes.

Calories are estimated using per-100g scaling from the Excel DB.

Requires well-authored JSON templates for each calorie band.

Variety depends on having multiple options in each slot.

Does not yet rebalance macronutrients beyond calories.

This documentation explains the system architecture and flow without needing to run code. It shows how the Excel nutrition database and JSON diet templates work together to create a personalized, realistic 4-week meal plan.
