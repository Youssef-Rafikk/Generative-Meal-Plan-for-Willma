import os, re, csv, json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass
import random


@dataclass
class Food:
    name: str
    type_name: str = ""
    sub_category: Optional[str] = None
    calories: float = 0.0
    protein: float = 0.0
    carbohydrates: float = 0.0
    fat: float = 0.0
    tags: str = ""
    order: int = 0

NUTRITION_DB: Dict[str, Food] = {
    # Animal proteins
    "chicken breast": Food("Chicken breast", type_name="protein", sub_category="poultry",
                           calories=165, protein=31, carbohydrates=0, fat=3.6,
                           tags="lean_protein, meat_product, cooked"),
    "beef steak": Food("Beef steak", type_name="protein", sub_category="red_meat",
                       calories=250, protein=26, carbohydrates=0, fat=17,
                       tags="high_protein, meat_product, cooked"),
    "tuna (canned)": Food("Tuna (canned)", type_name="protein", sub_category="fish",
                          calories=132, protein=29, carbohydrates=0, fat=1,
                          tags="lean_protein, fish_seafood, canned"),
    # Plant proteins
    "tofu (firm)": Food("Tofu (firm)", type_name="protein", sub_category="soy",
                        calories=76, protein=8, carbohydrates=2, fat=4.8,
                        tags="vegan, soy, high_protein, lean_protein, legume_bean_pea"),
    "cooked chickpeas": Food("Cooked chickpeas", type_name="carbohydrate", sub_category="legume",
                             calories=164, protein=8.9, carbohydrates=27.4, fat=2.6,
                             tags="vegan, legume_bean_pea, high_fiber"),
    "Cooked fava beans (foul medames)": Food("Cooked fava beans", type_name="carbohydrate", sub_category="legume",
                              calories=110, protein=7.6, carbohydrates=19.7, fat=0.4,
                              tags="vegan, legume_bean_pea, high_fiber"),
    # Dairy
    "greek yogurt": Food("Greek yogurt", type_name="protein", sub_category="dairy",
                         calories=97, protein=10, carbohydrates=3.6, fat=5.3,
                         tags="dairy_product, yogurt"),
    "coconut yogurt (unsweetened)": Food("Coconut yogurt (unsweetened)", type_name="fat", sub_category="plant_dairy_alt",
                                         calories=100, protein=1.5, carbohydrates=3, fat=8,
                                         tags="vegan, dairy_free, coconut, yogurt"),
    "oat milk (unsweetened)": Food("Oat milk (unsweetened)", type_name="carbohydrate", sub_category="plant_milk",
                                   calories=43, protein=1, carbohydrates=6.7, fat=1.4,
                                   tags="vegan, dairy_free, gluten",
    ),
    # Starches
    "cooked rice": Food("Cooked rice", type_name="carbohydrate", sub_category="grain",
                        calories=130, protein=2.4, carbohydrates=28.2, fat=0.3,
                        tags="grain, gluten_free"),
    "pasta (wheat, cooked)": Food("Pasta (wheat, cooked)", type_name="carbohydrate", sub_category="pasta",
                                  calories=158, protein=5.8, carbohydrates=30.9, fat=0.9,
                                  tags="grain, pasta, gluten"),
    # Fats
    "olive oil": Food("Olive oil", type_name="fat", sub_category="oil",
                      calories=884, protein=0, carbohydrates=0, fat=100,
                      tags="oil, fat, dressing_ingredient, vegan"),
}

def food_has_tag(food: Food, tag: str) -> bool:
    return tag.lower() in [t.strip().lower() for t in (food.tags or "").split(",")]

def food_has_any_tag(food: Food, tags: List[str]) -> bool:
    st = set(t.strip().lower() for t in (food.tags or "").split(",") if t.strip())
    return any(t.lower() in st for t in tags)

# ========================= Alternative Food Scorer =========================

class AlternativeFoodScorer:
    FUNCTIONAL_TAGS = set([
        "sweetener","flour","oil","thickener","lean_protein","high_protein","high_fiber",
        "gluten_free","vegan","dairy_free","baking_ingredient","dressing_ingredient","staple",
        "condiment","spice","spread","beverage","snack","cereal","powdered","grain","fruit_item",
        "vegetable_item","meat_product","dairy_product","fish_seafood","nut_based","seed_based",
        "legume_bean_pea","root_veg","leafy_green_veg","herb","pasta","bread","cheese","yogurt",
        "milk","juice","syrup","paste_form","dried","fresh","raw","cooked","liquid","solid","whole",
        "ground","binding_agent","leavening_agent","flavoring_agent","acidulant",
    ])

    def parse_tags(self, tags: Optional[str]) -> List[str]:
        if not tags:
            return []
        return [t.strip() for t in tags.split(",")
                if t.strip() and t.strip().lower() not in ("nan","na","-")]

    def get_primary_macro(self, food: Food) -> str:
        macros = {
            "protein": food.protein or 0,
            "carbohydrates": food.carbohydrates or 0,
            "fat": food.fat or 0,
        }
        return max(macros.items(), key=lambda kv: kv[1])[0]

    def calculate_equivalent_amount(self, original: Food, original_qty_g: float,
                                   alt: Food, primary_macro: str) -> float:
        target_macro_g = (getattr(original, primary_macro) / 100.0) * original_qty_g
        alt_per_g = (getattr(alt, primary_macro) / 100.0)
        if alt_per_g == 0:
            return 0.0
        return target_macro_g / alt_per_g

    def get_suitability(self, original: Food, alt: Food, original_qty_g: float,
                        alt_qty_g: float, primary_macro: str) -> str:
        if alt_qty_g > original_qty_g * 1.8:
            return "Unrecommended"
        sec = [m for m in ["protein","carbohydrates","fat"] if m != primary_macro]
        t1 = (getattr(original, sec[0]) / 100.0) * original_qty_g
        t2 = (getattr(original, sec[1]) / 100.0) * original_qty_g
        a1 = (getattr(alt, sec[0]) / 100.0) * alt_qty_g
        a2 = (getattr(alt, sec[1]) / 100.0) * alt_qty_g
        if abs(a1 - t1) > 5 or abs(a2 - t2) > 5:
            return "High Variance"
        return "Good Match"

    def compute_swap(self, original: Food, alt: Food, original_qty_g: float) -> Dict[str, Any]:
        primary = self.get_primary_macro(original)
        alt_qty = self.calculate_equivalent_amount(original, original_qty_g, alt, primary)
        if alt_qty <= 0:
            return {"ok": False}
        suitability = self.get_suitability(original, alt, original_qty_g, alt_qty, primary)
        macros = {
            "protein": round((alt.protein/100.0)*alt_qty, 1),
            "carbohydrates": round((alt.carbohydrates/100.0)*alt_qty, 1),
            "fat": round((alt.fat/100.0)*alt_qty, 1),
        }
        return {
            "ok": True,
            "quantity_g": round(alt_qty),
            "suitability": suitability,
            "macros": macros
        }

SCORER = AlternativeFoodScorer()

# ========================= Constants =========================

Calorie_Categories: List[int] = list(range(1200, 3001, 200))
Loss: int = 500
Gain: int = 300

ACTIVITY_LEVELS = {
    "1": ("sedentary",         1.2),
    "2": ("lightly_active",    1.375),
    "3": ("moderately_active", 1.55),
    "4": ("very_active",       1.725),
    "5": ("extra_active",      1.9),
}
HIGH_LEVEL_OPTIONS = ["Vegan", "Vegetarian", "Dairy-free", "Gluten-free"]
DISLIKE_GROUP_OPTIONS = [
    "None",
    "Seafood (fish & shellfish)",
    "Red meat (beef, lamb, etc.)",
    "Poultry (chicken, turkey)",
    "Pork",
    "Eggs",
    "Legumes (beans, lentils, peas)",
    "Soy products",
    "Nuts & seeds",
    "Cruciferous veg (broccoli, cauliflower, cabbage)",
    "Nightshades (tomato, pepper, eggplant)",
    "Starchy veg (potato, corn)",
    "Sweets & desserts",
    "Spicy foods",
]
UNIFIED_OPTIONS = ["None"] + HIGH_LEVEL_OPTIONS + DISLIKE_GROUP_OPTIONS[1:]
_CANON_SLOT_TITLES = {
    "meal 1": "Meal 1",
    "meal 2": "Meal 2",
    "meal 3": "Meal 3",
    "meal 4": "Meal 4",
    "meal 5": "Meal 5",
    "meal 6": "Meal 6",
    "snack": "Snack",
    "snack 1": "Snack 1",
    "snack 2": "Snack 2",
    "snack 3": "Snack 3",
}

# ========================= Input helpers =========================

def get_float(prompt: str, min_value: float = 0.0) -> float:
    while True:
        try:
            val = float(input(prompt).strip())
            if val <= min_value:
                print(f"Please enter a value greater than {min_value}")
                continue
            return val
        except ValueError:
            print("Please enter a valid number")

def get_int(prompt: str, min_value: int = 0) -> int:
    while True:
        try:
            val = int(input(prompt).strip())
            if val <= min_value:
                print(f"Please enter an integer greater than {min_value}")
                continue
            return val
        except ValueError:
            print("Please enter a valid integer")

def choose_from_list(options: List[str], title: str) -> int:
    while True:
        print(f"\n{title}")
        for i, opt in enumerate(options, start=1):
            print(f"  {i}) {opt}")
        choice = input("Enter the number of your choice: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
        print("Invalid selection. Please enter a valid number")

def choose_multi_from_list(options: List[str], title: str) -> List[int]:
    chosen: List[int] = []
    available = list(range(len(options)))
    while True:
        print(f"\n{title}")
        for j, idx in enumerate(available, start=1):
            print(f"  {j}) {options[idx]}")
        choice = input("Pick one option by number: ").strip()
        if choice.isdigit():
            pos = int(choice) - 1
            if 0 <= pos < len(available):
                picked_idx = available.pop(pos)
                if options[picked_idx].lower() == "none":
                    return []
                if picked_idx not in chosen:
                    chosen.append(picked_idx)
                again = input("Add another? (1=Yes, 2=No): ").strip()
                if again == "2":
                    break
                else:
                    continue
        print("Invalid selection. Please choose a valid number")
    return sorted(chosen)

# ========================= Calculation helpers =========================

def bmr(gender: str, weight_kg: float, height_cm: float, age_years: int) -> float:
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age_years
    return base + 5 if gender == "male" else base - 161

def TDEE(bmr: float, activity_factor: float) -> float:
    return bmr * activity_factor

def target_kcal(tdee: float, goal: str) -> Tuple[float, dict]:
    if goal == "lose":
        target = max(1200, tdee - Loss)
        return target, {"type": "deficit", "amount_kcal": Loss}
    else:
        target = tdee + Gain
        return target, {"type": "surplus", "amount_kcal": Gain}

def assign_category(kcal: float, goal: str) -> Tuple[int, str]:
    catigs = sorted(Calorie_Categories)
    if kcal <= catigs[0]:
        return catigs[0], f"Target ({kcal:.0f}) ≤ {catigs[0]} → assigned {catigs[0]}"
    if kcal >= catigs[-1]:
        return catigs[-1], f"Target ({kcal:.0f}) ≥ {catigs[-1]} → assigned {catigs[-1]}"
    for low, high in zip(catigs[:-1], catigs[1:]):
        if low < kcal < high:
            if goal == "lose":
                return low, f"Target ({kcal:.0f}) between {low} and {high}; goal is weight loss → round DOWN to {low}"
            else:
                return high, f"Target ({kcal:.0f}) between {low} and {high}; goal is mass gain → round UP to {high}"
    return catigs[0], "Fallback assignment"

# ========================= Preference keywords =========================

KEYS = {
    "meat_red": ["beef", "kofta", "lamb", "steak", "mince", "shawarma"],
    "poultry": ["chicken", "turkey", "breast", "thigh", "wings"],
    "seafood": ["fish", "tuna", "salmon", "shrimp", "prawn"],
    "pork": ["pork", "bacon", "ham"],
    "eggs": ["egg", "eggs"],
    "legumes": ["lentil", "lentils", "chickpea", "chickpeas", "beans", "bean", "ful", "fava"],
    "soy": ["soy", "tofu", "edamame", "tempeh"],
    "nuts": ["almond", "peanut", "tahini", "sesame", "nuts", "walnut", "cashew", "pistachio"],
    "crucifer": ["broccoli", "cauliflower", "cabbage"],
    "nightshade": ["tomato", "pepper", "eggplant"],
    "starchy": ["potato", "potatoes", "corn"],
    "sweets": ["honey", "dessert", "sweet", "candy", "sugar"],
    "spicy": ["shakshuka", "harissa", "spicy", "chili", "chilli", "hot sauce"],
    "dairy": ["yogurt", "yoghurt", "labneh", "cheese", "milk", "cottage cheese"],
    "gluten": ["bread", "toast", "pasta", "bulgur", "freekeh", "wheat", "barley"],
}

def contains_any(text: str, words: List[str]) -> bool:
    t = text.lower()
    for w in words:
        pattern = r'\b' + re.escape(w.lower()) + r's?\b'
        if re.search(pattern, t):
            return True
    return False

# ========================= Constraint checks =========================

def option_violates_prefs(opt: Dict[str, Any], allergies: List[str], dislikes: List[str]) -> bool:
    text = opt.get("option_name","")

    if "Vegan" in allergies:
        if contains_any(text, KEYS["dairy"] + KEYS["eggs"] + KEYS["meat_red"] + KEYS["poultry"] + KEYS["seafood"] + KEYS["pork"]):
            return True
    elif "Vegetarian" in allergies:
        if contains_any(text, KEYS["meat_red"] + KEYS["poultry"] + KEYS["seafood"] + KEYS["pork"]):
            return True
    if "Dairy-free" in allergies and contains_any(text, KEYS["dairy"]):
        return True
    if "Gluten-free" in allergies and contains_any(text, KEYS["gluten"]):
        return True

    mapping = {
        "Seafood (fish & shellfish)": KEYS["seafood"],
        "Red meat (beef, lamb, etc.)": KEYS["meat_red"],
        "Poultry (chicken, turkey)": KEYS["poultry"],
        "Pork": KEYS["pork"],
        "Eggs": KEYS["eggs"],
        "Legumes (beans, lentils, peas)": KEYS["legumes"],
        "Soy products": KEYS["soy"],
        "Nuts & seeds": KEYS["nuts"],
        "Cruciferous veg (broccoli, cauliflower, cabbage)": KEYS["crucifer"],
        "Nightshades (tomato, pepper, eggplant)": KEYS["nightshade"],
        "Starchy veg (potato, corn)": KEYS["starchy"],
        "Sweets & desserts": KEYS["sweets"],
        "Spicy foods": KEYS["spicy"],
    }
    for label, words in mapping.items():
        if label in dislikes and contains_any(text, words):
            return True
    return False

# ========================= Advanced replacement =========================

@dataclass
class Replacement:
    name: str
    grams_hint: Optional[str] = None
    vegan: bool = True
    vegetarian: bool = True
    soy: bool = False
    dairy: bool = False
    gluten: bool = False
    note: str = ""

PROTEIN_REPLACEMENTS = [
    Replacement("Tofu (firm)", "150 g", soy=True),
    Replacement("Cooked fava beans (foul medames)", "200 g"),
    Replacement("Cooked chickpeas", "200 g"),
    Replacement("Grilled eggplant slices", "200 g"),
]
EGG_REPLACEMENTS = [
    Replacement("Chickpea flour omelet", "2/3 cup batter"),
    Replacement("Mashed fava beans scramble", "180 g"),
]
DAIRY_REPLACEMENTS = [
    Replacement("Coconut yogurt (unsweetened)", "170 g"),
    Replacement("Almond yogurt (unsweetened)", "170 g"),
    Replacement("Oat milk (unsweetened)", "250 ml", gluten=True),
    Replacement("Soy yogurt (unsweetened)", "170 g", soy=True),
    Replacement("Rice milk (unsweetened)", "250 ml"),
]
GLUTEN_REPLACEMENTS = [
    Replacement("Cooked rice", "150 g"),
    Replacement("Cooked freekeh", "150 g", gluten=True),
    Replacement("Cooked quinoa", "150 g"),
    Replacement("Corn tortillas or baladi-style corn flatbread", "2 pcs"),
    Replacement("Gluten-free pasta (imported)", "75 g dry"),
]
NUTS_SEEDS_REPLACEMENTS = [
    Replacement("Hummus without tahini", "100 g"),
    Replacement("Avocado", "100 g"),
    Replacement("Green olives", "40 g"),
    Replacement("Black olives", "40 g"),
]
SPICY_REPLACEMENTS = [
    Replacement("Tomato-free mild herb dip", "2 tbsp"),
    Replacement("Za'atar herb mix", None),
]
SWEETS_REPLACEMENTS = [
    Replacement("Fresh dates", "2–3 pcs / ~50 g"),
    Replacement("Seasonal fresh fruit", "150 g"),
    Replacement("85% dark chocolate", "15 g"),
]
STARCHY_VEG_REPLACEMENTS = [
    Replacement("Roasted sweet potato", "200 g"),
    Replacement("Pumpkin", "200 g"),
    Replacement("Carrots", "200 g"),
]
CRUCIFEROUS_REPLACEMENTS = [
    Replacement("Cauliflower florets (steamed)", "200 g"),
    Replacement("Broccoli florets (steamed)", "200 g"),
    Replacement("Green beans", "200 g"),
]
NIGHTSHADE_REPLACEMENTS = [
    Replacement("Cucumber", "150 g"),
    Replacement("Zucchini", "150 g"),
]

REPLACEMENT_POOLS = {
    "meat_red": PROTEIN_REPLACEMENTS,
    "poultry": PROTEIN_REPLACEMENTS,
    "seafood": PROTEIN_REPLACEMENTS,
    "pork":    PROTEIN_REPLACEMENTS,
    "eggs":    EGG_REPLACEMENTS,
    "dairy":   DAIRY_REPLACEMENTS,
    "gluten":  GLUTEN_REPLACEMENTS,
    "nuts":    NUTS_SEEDS_REPLACEMENTS,
    "spicy":   SPICY_REPLACEMENTS,
    "sweets":  SWEETS_REPLACEMENTS,
    "starchy": STARCHY_VEG_REPLACEMENTS,
    "crucifer": CRUCIFEROUS_REPLACEMENTS,
    "nightshade": NIGHTSHADE_REPLACEMENTS,
}

def _is_constraint_active(flag: str, allergies: List[str], dislikes: List[str]) -> bool:
    if flag == "vegan":        return "Vegan" in allergies
    if flag == "vegetarian":   return "Vegetarian" in allergies
    if flag == "dairy_free":   return "Dairy-free" in allergies
    if flag == "gluten_free":  return "Gluten-free" in allergies
    if flag == "no_soy":       return "Soy products" in dislikes
    if flag == "no_legumes":   return "Legumes (beans, lentils, peas)" in dislikes
    if flag == "no_nuts":      return "Nuts & seeds" in dislikes
    return False

def _allowed_replacement(rep: Replacement, allergies: List[str], dislikes: List[str]) -> bool:
    if _is_constraint_active("vegan", allergies, dislikes) and not rep.vegan:
        return False
    if _is_constraint_active("vegetarian", allergies, dislikes) and not rep.vegetarian:
        return False
    if _is_constraint_active("no_soy", allergies, dislikes) and rep.soy:
        return False
    if _is_constraint_active("dairy_free", allergies, dislikes) and rep.dairy:
        return False
    if _is_constraint_active("gluten_free", allergies, dislikes) and rep.gluten:
        return False
    legume_tokens = ("lentil", "chickpea", "bean", "beans", "fava", "ful", "hummus")
    if _is_constraint_active("no_legumes", allergies, dislikes) and any(tok in rep.name.lower() for tok in legume_tokens):
        return False
    nut_tokens = ("almond", "peanut", "cashew", "walnut", "pistachio", "tahini", "sesame", "nuts")
    if _is_constraint_active("no_nuts", allergies, dislikes) and any(tok in rep.name.lower() for tok in nut_tokens):
        return False
    return True

def _pick_replacement_for_category_fallback(cat: str, allergies: List[str], dislikes: List[str]) -> Optional[Replacement]:
    pool = REPLACEMENT_POOLS.get(cat, [])
    allowed = [rep for rep in pool if _allowed_replacement(rep, allergies, dislikes)]
    if not allowed:
        return None
    return random.choice(allowed)

def _token_replace(text: str, targets: List[str], replacement_text: str) -> str:
    if not targets:
        return text
    lead_qty = (
        r"(?:\b\d+(?:\.\d+)?\s*"
        r"(?:g|kg|ml|l|cup|cups|tbsp|tablespoons?|tsp|teaspoons?|"
        r"slice|slices|piece|pieces|pc|pcs|whole|eggs?)?"
        r"\s+)?"
    )
    qualifiers_before = r"(?:lean|skinless|boneless|low[- ]fat|full[- ]fat|fat[- ]free|reduced[- ]fat|skim|cottage|greek)\s+"
    cuts_after = (
        r"(?:breast|thigh|drumstick|wing|wings|fillet|filet|steak|mince|"
        r"tenderloin|sirloin|ribeye|cutlet|white|whites|yolk|yolks)s?"
    )
    target_alt = "|".join(sorted({re.escape(w.lower()) for w in targets}, key=len, reverse=True))
    pattern = rf"{lead_qty}(?:{qualifiers_before})?(?:{target_alt})(?:\s+{cuts_after})?\b"
    placeholder = "\uFFFFREPL\uFFFF"
    temp = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    out = temp.replace(placeholder, replacement_text)
    out = re.sub(r"\s*\+\s*", " + ", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" .")
    return out

def _cleanup_artifacts(text: str) -> str:
    text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\+\s*", " + ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" .")
    return text

def _allowed_food_by_constraints(food: Food, allergies: List[str], dislikes: List[str]) -> bool:
    if "Vegan" in allergies and not food_has_any_tag(food, ["vegan","dairy_free"]):
        if food_has_any_tag(food, ["meat_product","fish_seafood","dairy_product","egg"]):
            return False
    if "Vegetarian" in allergies and food_has_any_tag(food, ["meat_product","fish_seafood"]):
        return False
    if "Dairy-free" in allergies and food_has_any_tag(food, ["dairy_product"]):
        return False
    if "Gluten-free" in allergies and food_has_any_tag(food, ["gluten","wheat","bread","pasta"]):
        return False
    if "Soy products" in dislikes and ("soy" in (food.tags or "").lower()):
        return False
    if "Legumes (beans, lentils, peas)" in dislikes and food_has_any_tag(food, ["legume_bean_pea"]):
        return False
    if "Nuts & seeds" in dislikes and food_has_any_tag(food, ["nut_based","seed_based"]):
        return False
    return True

def _find_original_foods_in_text(option_text: str) -> List[Food]:
    text = option_text.lower()
    hits = []
    for name, food in NUTRITION_DB.items():
        pat = r"\b" + re.escape(name) + r"\b"
        if re.search(pat, text):
            hits.append(food)
    return hits

def _candidate_alternatives_for(original: Food, allergies: List[str], dislikes: List[str]) -> List[Food]:
    orig_tags = set(t.lower() for t in (original.tags or "").split(",") if t.strip())
    candidates = []
    for f in NUTRITION_DB.values():
        if f.name == original.name:
            continue
        if not _allowed_food_by_constraints(f, allergies, dislikes):
            continue
        share_sub = (original.sub_category and f.sub_category and original.sub_category == f.sub_category)
        share_type = (original.type_name and f.type_name and original.type_name.lower() == f.type_name.lower())
        f_tags = set(t.lower() for t in (f.tags or "").split(",") if t.strip())
        share_func = len((orig_tags & f_tags) & AlternativeFoodScorer.FUNCTIONAL_TAGS) > 0
        if share_sub or share_type or share_func:
            candidates.append(f)
    return candidates

def _scored_swap_text(original: Food, offenders_tokens: List[str],
                      allergies: List[str], dislikes: List[str],
                      assumed_original_qty_g: int = 150) -> Optional[str]:
    cands = _candidate_alternatives_for(original, allergies, dislikes)
    best = None
    best_rank = (99, 999999)
    tier_index = {"Good Match": 0, "High Variance": 1, "Unrecommended": 2}
    for alt in cands:
        result = SCORER.compute_swap(original, alt, assumed_original_qty_g)
        if not result["ok"]:
            continue
        tier = tier_index[result["suitability"]]
        rank = (tier, alt.order or 0)
        if rank < best_rank:
            best_rank = rank
            best = (alt, result)
    if not best:
        return None
    alt, swap = best
    qty = swap["quantity_g"]
    return f"{alt.name} {qty} g"

def _dedupe_plus_items(text: str) -> str:
    parts = [p.strip(" .") for p in re.split(r'\s*\+\s*', text) if p.strip(" .")]
    seen = set()
    out = []
    for p in parts:
        key = re.sub(r"\s+", " ", p).strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return " + ".join(out)

def _parse_option_items_freeform(option_text: str) -> List[Dict[str, Any]]:
    parts = [p.strip(" .") for p in option_text.split("+")]
    return [{"food": p, "grams": None} for p in parts if p]

def rewrite_option_text_for_prefs(option_text: str, allergies: List[str], dislikes: List[str]) -> str:
    new_text = option_text
    active_categories = []
    if "Vegan" in allergies:
        active_categories += ["dairy", "eggs", "meat_red", "poultry", "seafood", "pork"]
    elif "Vegetarian" in allergies:
        active_categories += ["meat_red", "poultry", "seafood", "pork"]
    if "Dairy-free" in allergies:
        active_categories.append("dairy")
    if "Gluten-free" in allergies:
        active_categories.append("gluten")

    dislike_map = {
        "Seafood (fish & shellfish)": "seafood",
        "Red meat (beef, lamb, etc.)": "meat_red",
        "Poultry (chicken, turkey)": "poultry",
        "Pork": "pork",
        "Eggs": "eggs",
        "Legumes (beans, lentils, peas)": "legumes",
        "Soy products": "soy",
        "Nuts & seeds": "nuts",
        "Cruciferous veg (broccoli, cauliflower, cabbage)": "crucifer",
        "Nightshades (tomato, pepper, eggplant)": "nightshade",
        "Starchy veg (potato, corn)": "starchy",
        "Sweets & desserts": "sweets",
        "Spicy foods": "spicy",
    }
    for label in dislikes:
        if label in dislike_map:
            active_categories.append(dislike_map[label])

    seen = set()
    active_categories = [c for c in active_categories if not (c in seen or seen.add(c))]
    originals_in_text = _find_original_foods_in_text(new_text)
    for original in originals_in_text:
        disallowed = False
        offender_tokens: List[str] = []
        if "Vegan" in allergies and food_has_any_tag(original, ["meat_product","fish_seafood","dairy_product","egg"]):
            disallowed = True
        elif "Vegetarian" in allergies and food_has_any_tag(original, ["meat_product","fish_seafood"]):
            disallowed = True
        elif "Dairy-free" in allergies and food_has_any_tag(original, ["dairy_product"]):
            disallowed = True
        elif "Gluten-free" in allergies and food_has_any_tag(original, ["gluten"]):
            disallowed = True
        if disallowed:
            for cat, words in KEYS.items():
                if contains_any(original.name, words) or contains_any(new_text, words):
                    offender_tokens = words
                    break
            replacement_text = _scored_swap_text(
                original,
                offender_tokens or [original.name.lower()],
                allergies, dislikes,
                assumed_original_qty_g=150
            )
            if replacement_text:
                targets = offender_tokens or [original.name.lower()]
                new_text = _token_replace(new_text, targets, replacement_text)

    for cat in active_categories:
        words = KEYS.get(cat, [])
        if cat == "eggs":
            words = ["egg", "eggs", "egg white", "egg whites", "egg yolk", "egg yolks"]
        if not words:
            continue
        if contains_any(new_text, words):
            rep = _pick_replacement_for_category_fallback(cat, allergies, dislikes)
            if rep:
                rtxt = rep.name + (f" {rep.grams_hint}" if rep.grams_hint else "")
                new_text = _token_replace(new_text, words, rtxt)
            else:
                for w in words:
                    pattern = r'\b' + re.escape(w) + r's?\b'
                    new_text = re.sub(pattern, "", new_text, flags=re.IGNORECASE)
                new_text = re.sub(r"\s{2,}", " ", new_text).strip(" +.-")

    new_text = _cleanup_artifacts(new_text)
    new_text = _dedupe_plus_items(new_text)
    return new_text

def apply_replacements_to_diet_options(diet: Dict[str, Any], allergies: List[str], dislikes: List[str]) -> Dict[str, Any]:
    new_d = dict(diet)
    new_slots: Dict[str, List[Dict[str, Any]]] = {}
    for slot, opts in diet["slots"].items():
        new_opts = []
        for opt in opts:
            txt = opt.get("option_name", "")
            fixed = rewrite_option_text_for_prefs(txt, allergies, dislikes)
            new_opt = dict(opt)
            new_opt["option_name"] = fixed
            new_opt["items"] = _parse_option_items_freeform(fixed)
            new_opts.append(new_opt)
        new_slots[slot] = new_opts
    new_d["slots"] = new_slots
    return new_d

# ========================= JSON band loader (NEW) =========================

def _slot_key_from_number(n: int) -> str:
    return f"meal {int(n)}"

def _make_slot_sequence(mpd: int) -> List[str]:
    return [_slot_key_from_number(i) for i in range(1, mpd + 1)]

def compute_band_filename_from_target(target_kcal: float) -> str:
    target_int = int(round(target_kcal))
    if target_int >= 3000:
        return "3000+_kcal.json"
    if target_int < 1200:
        target_int = 1200
    low = (target_int // 200) * 200
    high = low + 200
    return f"{low}_{high}_kcal.json"

def _infer_kcal_range_from_filename(fname: str) -> Tuple[Optional[int], Optional[int]]:
    stem = Path(fname).stem
    if stem.startswith("3000+"):
        return (3000, None)
    try:
        parts = stem.split("_")
        low = int(parts[0]); high = int(parts[1])
        return (low, high)
    except Exception:
        return (None, None)

def _slot_key_from_json_meal_number(mnum, counters: Dict[str, int]) -> str:
    """Normalize JSON 'meal_number' into our internal slot key, preserving snacks."""
    if isinstance(mnum, int):
        return _slot_key_from_number(mnum)
    s = str(mnum).strip().lower()
    if s == "snack":
        n = counters.get("snack", 0) + 1
        counters["snack"] = n
        return "snack" if n == 1 else f"snack {n}"
    n = counters.get(s, 0) + 1
    counters[s] = n
    return s if n == 1 else f"{s} {n}"

def _convert_json_structure_to_internal(json_diet: Dict[str, Any],
                                        kcal_range: Tuple[Optional[int], Optional[int]]
                                        ) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    # Case A: already internal
    if "slots" in json_diet and "slot_sequence" in json_diet:
        item = {
            "diet_label": json_diet.get("diet_label", ""),
            "diet_kcal": json_diet.get("diet_kcal") or json_diet.get("total_kcal"),
            "kcal_range": kcal_range,
            "structure": json_diet.get("structure") or {},
            "slot_sequence": json_diet.get("slot_sequence"),
            "slots": json_diet["slots"],
        }
        if not item["structure"]:
            total_slots = len(item["slot_sequence"])
            item["structure"] = {"meals": total_slots, "snacks": 0, "total_slots": total_slots}
        results.append(item)
        return results

    # Case B: convert from structures -> internal (preserve Snack position)
    diet_label = json_diet.get("diet_label", "")
    diet_kcal  = json_diet.get("diet_kcal") or json_diet.get("total_kcal")
    structures = json_diet.get("structures", [])
    for st in structures:
        mpd = st.get("meals_per_day")
        if not mpd:
            continue

        slots: Dict[str, List[Dict[str, Any]]] = OrderedDict()
        slot_sequence: List[str] = []
        counters: Dict[str, int] = {}

        for meal in st.get("meals", []):
            mnum = meal.get("meal_number")
            key = _slot_key_from_json_meal_number(mnum, counters)
            options = meal.get("options", [])
            built_opts = []
            for idx, opt in enumerate(options, start=1):
                raw_items = opt.get("items", [])
                if isinstance(raw_items, list):
                    option_name = " + ".join(raw_items).strip(" .")
                    items = [{"food": s, "grams": None} for s in raw_items]
                else:
                    option_name = str(raw_items)
                    items = _parse_option_items_freeform(option_name)
                built_opts.append({
                    "slot": key,
                    "option_ref": f"Option {idx}",
                    "option_name": option_name,
                    "items": items
                })

            if key not in slots:
                slot_sequence.append(key)
                slots[key] = built_opts
            else:
                slots[key].extend(built_opts)

        if not slot_sequence:
            slot_sequence = _make_slot_sequence(mpd)
            for key in slot_sequence:
                slots.setdefault(key, [])

        structure = {"meals": mpd, "snacks": 0, "total_slots": mpd}
        results.append({
            "diet_label": diet_label,
            "diet_kcal": diet_kcal,
            "kcal_range": kcal_range,
            "structure": structure,
            "slot_sequence": slot_sequence,
            "slots": slots,
        })
    return results

def load_diets_from_band(json_dir: str, filename: str) -> List[Dict[str, Any]]:
    path = Path(json_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Band file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    diets_json = data.get("diets") if isinstance(data, dict) else data
    if not isinstance(diets_json, list):
        raise ValueError(f"Invalid JSON structure in {path}: expected a list or an object with 'diets'.")
    kcal_range = _infer_kcal_range_from_filename(filename)
    all_internal: List[Dict[str, Any]] = []
    for jd in diets_json:
        all_internal.extend(_convert_json_structure_to_internal(jd, kcal_range))
    if not all_internal:
        raise ValueError(f"No diets found after conversion in {path}")
    return all_internal

# ========================= Plan building =========================

def _slot_is_meal(slot: str) -> bool:
    return slot.lower().startswith("meal")

def _nonempty_slots(diet: Dict[str, Any]) -> List[str]:
    return [s for s in diet.get("slot_sequence", []) if len(diet["slots"].get(s, [])) > 0]

def _same_option_set(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    if a is None or b is None:
        return False
    names_a = tuple(sorted((opt.get("option_name","") for opt in a)))
    names_b = tuple(sorted((opt.get("option_name","") for opt in b)))
    return names_a == names_b and len(names_a) > 0

def _pick_option_index(
    rng: random.Random,
    opts: List[Dict[str, Any]],
    avoid_names: Optional[List[str]] = None,
    avoid_prob: float = 0.0
) -> int:

    n = len(opts)
    if n <= 1:
        return 0
    avoid_names = [n for n in (avoid_names or []) if n]  # clean
    use_exclusion = (avoid_names and rng.random() < avoid_prob)
    candidates = list(range(n))
    if use_exclusion:
        candidates = [i for i in candidates if opts[i].get("option_name") not in avoid_names] or list(range(n))
    return rng.choice(candidates)

def select_diets_for_user(
    diets: List[Dict[str, Any]],
    assigned_category: int,
    meals_per_day: int,
    goal: str,
) -> List[Dict[str, Any]]:
    viable = [d for d in diets if sum(1 for s in d.get("slot_sequence", []) if len(d["slots"].get(s, [])) > 0) >= meals_per_day]
    base_pool = viable if viable else diets[:]

    def by_structure(pool: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        exact = []
        others = []
        for d in pool:
            total = (d.get("structure") or {}).get("total_slots")
            if total == meals_per_day:
                exact.append(d)
            else:
                others.append(d)
        return exact, others

    exact_pool, other_pool = by_structure(base_pool)

    def apply_goal_and_window(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pool:
            return []
        dir_filtered: List[Dict[str, Any]] = []
        for d in pool:
            dk = d.get("diet_kcal")
            if dk is None:
                dir_filtered.append(d)
                continue
            if goal == "lose":
                if dk <= assigned_category:
                    dir_filtered.append(d)
            else:
                if dk >= assigned_category:
                    dir_filtered.append(d)
        if not dir_filtered:
            dir_filtered = pool

        window_filtered: List[Dict[str, Any]] = []
        lower, upper = (assigned_category - 200, assigned_category) if goal == "lose" else (assigned_category, assigned_category + 200)
        for d in dir_filtered:
            dk = d.get("diet_kcal")
            if dk is None or (lower <= dk <= upper):
                window_filtered.append(d)
        result = window_filtered if window_filtered else dir_filtered
        result.sort(key=lambda d: abs((d.get("diet_kcal") or assigned_category) - assigned_category))
        return result

    chosen = apply_goal_and_window(exact_pool)
    if not chosen:
        chosen = apply_goal_and_window(other_pool)
    if not chosen:
        chosen = apply_goal_and_window(base_pool)
    return chosen

def build_one_day_from_diet(
    diet: Dict[str, Any],
    day_index: int,
    desired_slots: Optional[int] = None,
    rng: Optional[random.Random] = None,
    previous_day_choices: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    sequence_all = _nonempty_slots(diet)
    if not sequence_all:
        raise ValueError(f"Diet '{diet.get('diet_label','')}' has no available options in any slot.")
    sequence = sequence_all[:desired_slots] if desired_slots is not None else sequence_all[:]
    if rng is None:
        rng = random.Random()

    chosen: Dict[str, Dict[str, Any]] = {}
    prev_slot_opts: Optional[List[Dict[str, Any]]] = None
    prev_slot_pick_name: Optional[str] = None

    for slot in sequence:
        opts = diet["slots"].get(slot, [])
        if not opts:
            continue

        avoid_names: List[str] = []

        if _same_option_set(prev_slot_opts, opts) and prev_slot_pick_name:
            avoid_names.append(prev_slot_pick_name)
            avoid_prob = 0.60
        else:
            avoid_prob = 0.0

        if previous_day_choices:
            y_choice = previous_day_choices.get(slot)
            if y_choice:
                avoid_names.append(y_choice)
                avoid_prob = max(avoid_prob, 0.60)

        idx = _pick_option_index(rng, opts, avoid_names=avoid_names, avoid_prob=avoid_prob)
        pick = opts[idx]
        chosen[slot] = pick
        prev_slot_opts = opts
        prev_slot_pick_name = pick.get("option_name")

    if not chosen:
        raise ValueError(f"No schedulable options found for requested slots in diet '{diet.get('diet_label','')}'.")
    return {
        "diet_label": diet.get("diet_label"),
        "diet_kcal": diet.get("diet_kcal"),
        "slots": chosen
    }

def build_plan_from_diets(
    diets: List[Dict[str, Any]],
    allergies: List[str],
    dislikes: List[str],
    days: int,
    meals_per_day: Optional[int] = None,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    filtered_diets = [filter_diet_options_by_prefs(d, allergies, dislikes) for d in diets]
    last_choice_per_slot: Dict[str, str] = {}

    for d in range(days):
        diet = filtered_diets[d % len(filtered_diets)]
        day_plan = build_one_day_from_diet(
            diet,
            d,
            desired_slots=meals_per_day,
            rng=rng,
            previous_day_choices=last_choice_per_slot
        )
        plan.append(day_plan)
        for slot, opt in day_plan["slots"].items():
            last_choice_per_slot[slot] = opt.get("option_name")

    return plan

# ========================= Export helpers (JSON outputs) =========================

def _items_from_option_name(option_name: str) -> List[str]:
    # Split by " + " and clean
    parts = [p.strip(" .") for p in re.split(r'\s*\+\s*', option_name) if p.strip(" .")]
    return parts

def _label_range_str(low: Optional[int], high: Optional[int]) -> str:
    if low and high:
        return f"{low}–{high}"
    if low and not high:
        return f"{low}+"
    return ""

def _structures_from_internal_diet(internal: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert our internal normalized diet back into the requested 'structures' schema.
    We create exactly ONE structure per internal diet (since the loader already split per structure).
    """
    structure = internal.get("structure") or {}
    meals_per_day = structure.get("total_slots") or structure.get("meals") or len(_nonempty_slots(internal))
    meals: List[Dict[str, Any]] = []

    counters: Dict[str, int] = {"meal": 0, "snack": 0}

    for slot in internal.get("slot_sequence", []):
        opts = internal["slots"].get(slot, [])
        if not opts:
            continue

        # Decide meal_number label
        if slot.lower().startswith("snack"):
            meal_number: Any = "Snack"
            counters["snack"] += 1
            if counters["snack"] > 1:
                meal_number = "Snack"  # keep generic label per your example
        else:
            counters["meal"] += 1
            meal_number = counters["meal"]

        meal_entry = {
            "meal_number": meal_number,
            "options": [
                {"items": _items_from_option_name(o.get("option_name",""))}
                for o in opts
                if (o.get("option_name","") or "").strip()
            ]
        }
        meals.append(meal_entry)

    return [{
        "meals_per_day": meals_per_day,
        "meals": meals
    }]

def export_diets_structures_json(
    diets: List[Dict[str, Any]],
    label_range_kcal: str,
    outfile: str = "4_Week_Meal_Plan.json"
) -> None:
    """
    Writes a JSON file matching the requested schema:

    {
      "diets": [
        {
          "label_range_kcal": "2400–2600",
          "diet_label": "...",
          "total_kcal": 2533,
          "macros": {},  # not computed here
          "structures": [ { "meals_per_day": ..., "meals": [ ... ] } ]
        }, ...
      ]
    }
    """
    payload = {"diets": []}
    for d in diets:
        item = {
            "label_range_kcal": label_range_kcal,
            "diet_label": d.get("diet_label") or "",
            "total_kcal": d.get("diet_kcal"),
            "macros": {},  # placeholder; macros per diet not computed in this pipeline
            "structures": _structures_from_internal_diet(d)
        }
        payload["diets"].append(item)

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def export_user_profile_json(profile: Dict[str, Any], outfile: str = "user_profile.json") -> None:
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

# ========================= Unified constraint collection =========================

def collect_constraints_unified() -> Tuple[List[str], List[str]]:
    idxs = choose_multi_from_list(UNIFIED_OPTIONS, "Select diet/allergies & disliked foods:")
    if not idxs:
        return [], []
    names = [UNIFIED_OPTIONS[i] for i in idxs]
    allergies = [n for n in names if n in HIGH_LEVEL_OPTIONS]
    dislikes  = [n for n in names if n not in HIGH_LEVEL_OPTIONS and n != "None"]
    if "Vegan" in allergies and "Vegetarian" in allergies:
        allergies = [a for a in allergies if a != "Vegetarian"]
    def dedup_keep_order(items: List[str]) -> List[str]:
        seen = set(); out=[]
        for x in items:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    allergies = dedup_keep_order(allergies)
    dislikes  = dedup_keep_order(dislikes)
    return allergies, dislikes

# ========================= Preference filtering  =========================

def filter_diet_options_by_prefs(diet: Dict[str, Any], allergies: List[str], dislikes: List[str]) -> Dict[str, Any]:
    rewritten = apply_replacements_to_diet_options(diet, allergies, dislikes)
    cleaned_slots = {}
    for slot, opts in rewritten["slots"].items():
        kept = [o for o in opts if not option_violates_prefs(o, allergies, dislikes)]
        cleaned_slots[slot] = kept if kept else opts
    rewritten["slots"] = cleaned_slots
    return rewritten

# ========================= Interactive runner (JSON-first) =========================

def run_planner(
    json_dir: str = "JSON",
    weeks: int = 4,
    out_json_plan: str = "4_Week_Meal_Plan.json",
    user_profile_json: str = "user_profile.json",
    seed: Optional[int] = None  # set an int for reproducible randomness
):
    print("=== Meal Planner (JSON Templates → 4-Week Plan) ===\n")

    gender = ["male","female"][choose_from_list(["Male","Female"], "Select your gender:")]
    height_cm = get_float("Enter height (cm): ")
    weight_kg = get_float("Enter weight (kg): ")
    age = get_int("Enter age (years): ")
    goal = ["lose","gain"][choose_from_list(["Lose weight","Mass gain"], "Select your diet goal:")]

    activity_idx = choose_from_list(
        ["Sedentary (little/no exercise)",
         "Lightly active (1–3 days/week)",
         "Moderately active (3–5 days/week)",
         "Very active (6–7 days/week)",
         "Extra active (hard exercise/physical)"],
        "Select your activity level:"
    )
    activity_label, activity_factor = ACTIVITY_LEVELS[str(activity_idx + 1)]

    meals_per_day = 3 + choose_from_list(["3 meals/day", "4 meals/day", "5 meals/day", "6 meals/day"],
                                         "How many meals per day do you prefer?")

    allergies, dislikes = collect_constraints_unified()

    bmr_val = bmr(gender, weight_kg, height_cm, age)
    tdee_val = TDEE(bmr_val, activity_factor)
    target_kcal_val, target_meta = target_kcal(tdee_val, goal)
    assigned_category, reason = assign_category(target_kcal_val, goal)

    band_file = compute_band_filename_from_target(target_kcal_val)
    band_low, band_high = _infer_kcal_range_from_filename(band_file)
    label_range_kcal = _label_range_str(band_low, band_high)

    print("\n--- Calculations ---")
    print(f"BMR:                {bmr_val:.2f} kcal/day")
    print(f"TDEE:               {tdee_val:.2f} kcal/day")
    sign = "-" if goal == "lose" else "+"
    adj  = Loss if goal == "lose" else Gain
    print(f"Goal adjustment:    {sign}{adj} kcal → Target = {target_kcal_val:.0f} kcal/day")
    print(f"Assigned category:  {assigned_category} kcal  ({reason})")
    if band_high:
        print(f"JSON band file:     {band_file}  (band {band_low}–{band_high} kcal)")
    else:
        print(f"JSON band file:     {band_file}  (band {band_low}+ kcal)")

    if allergies:
        print("Allergies/diet:     " + ", ".join(allergies))
    if dislikes:
        print("Dislikes:           " + ", ".join(dislikes))

    diets = load_diets_from_band(json_dir, band_file)
    if not diets:
        raise ValueError(f"No diets parsed from the JSON file {band_file}.")

    days = weeks * 7

    chosen_diets = select_diets_for_user(
        diets=diets,
        assigned_category=assigned_category,
        meals_per_day=meals_per_day,
        goal=goal,
    )

    if band_low:
        print(f"\n[Info] Using calorie band {band_low}–{band_high or 'plus'} kcal (goal: {goal}, category: {assigned_category}).")

    rng_seed = seed
    plan = build_plan_from_diets(
        chosen_diets,
        allergies,
        dislikes,
        days=days,
        meals_per_day=meals_per_day,
        seed=rng_seed  # reproducible if provided
    )

    # ---------- Write user profile JSON ----------
    user_profile = {
        "inputs": {
            "gender": gender,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "age": age,
            "goal": goal,
            "activity_label": activity_label,
            "activity_factor": activity_factor,
            "meals_per_day": meals_per_day,
            "allergies": allergies,
            "dislikes": dislikes
        },
        "calculations": {
            "bmr": round(bmr_val, 2),
            "tdee": round(tdee_val, 2),
            "target_kcal": int(round(target_kcal_val)),
            "assigned_category": assigned_category,
            "category_reason": reason,
            "calorie_band_low": band_low,
            "calorie_band_high": band_high,
            "templates_json": band_file,
            "label_range_kcal": label_range_kcal,
            "weeks": weeks,
            "days": days,
            "seed": seed
        }
    }
    export_user_profile_json(user_profile, outfile=user_profile_json)
    print(f"[Saved] User profile → {user_profile_json}")


    filtered_for_export = [filter_diet_options_by_prefs(d, allergies, dislikes) for d in chosen_diets]
    export_diets_structures_json(filtered_for_export, label_range_kcal=label_range_kcal, outfile=out_json_plan)
    print(f"[Saved] 4-week diet set (structures schema) → {out_json_plan}")

    # ---------- print the 4-week schedule to console ----------
    print("\n--- 4-Week Plan ---")
    for w in range(weeks):
        print(f"\nWeek {w+1}")
        print("-" * 8)
        for d in range(7):
            idx = w*7 + d
            day = plan[idx]
            kcal = day.get("diet_kcal")
            kcal_str = f"~{kcal} kcal" if kcal else "—"
            print(f" Day {d+1}: {kcal_str}  |  {day.get('diet_label','')}")
            for slot in day["slots"].keys():
                opt = day["slots"][slot]
                slot_title = _CANON_SLOT_TITLES.get(slot, slot.title())
                print(f"   - {slot_title}: {opt['option_name']}")

    print("\n---------------------------\n")

# ========================= Main =========================

def main():
    run_planner(
        json_dir="JSON",
        weeks=4,
        out_json_plan="4_Week_Meal_Plan.json",
        user_profile_json="user_profile.json"
    )

if __name__ == "__main__":
    main()
