import os, re, json, glob
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass
import random
import pandas as pd


# ========================= Excel path resolver =========================

def resolve_excel_path(explicit_path: Optional[str] = None, filename_hint: str = "WILLMA Nutrtion Ingredients Database V5.0 .xlsx") -> Path:
    # 1) explicit path
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if p.exists():
            return p

    # 2) environment variable
    env_path = os.environ.get("WILLMA_FOOD_DB")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p

    # 3) common locations
    script_dir = Path(__file__).parent.resolve()
    cwd = Path.cwd().resolve()
    home = Path.home()
    common_dirs = [
        script_dir,
        cwd,
        script_dir / "data",
        cwd / "data",
        script_dir / "JSON",
        cwd / "JSON",
        home / "Desktop",
        home / "Downloads",
        home / "Documents"
    ]
    for d in common_dirs:
        p = d / filename_hint
        if p.exists():
            return p

    # 4) wildcard search near script and home
    patterns = [
        str(script_dir / "**" / "*.xlsx"),
        str(home / "**" / "*.xlsx")
    ]
    for pat in patterns:
        for hit in glob.glob(pat, recursive=True):
            hp = Path(hit)
            name_lower = hp.name.lower()
            if "willma" in name_lower and "nutr" in name_lower and "ingredient" in name_lower and name_lower.endswith(".xlsx"):
                return hp.resolve()

    # 5) file dialog picker as last resort
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        picked = filedialog.askopenfilename(
            title="Select WILLMA Nutrition Ingredients Database Excel file",
            filetypes=[("Excel", "*.xlsx *.xls")],
        )
        root.destroy()
        if picked:
            p = Path(picked).expanduser().resolve()
            if p.exists():
                return p
    except Exception:
        pass

    lines = [
        r"  EXCEL_DB_PATH = r'C:\Users\YOURNAME\Desktop\WILLMA Nutrtion Ingredients Database V5.0 .xlsx'",
    ]
    raise FileNotFoundError("\n".join(lines))


# ========================= Data model =========================

@dataclass(frozen=True)
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


# ========================= Excel-backed food DB =========================

EXCEL_DB_PATH = None  # e.g. r"C:\Users\express\Desktop\WILLMA Nutrtion Ingredients Database V5.0 .xlsx"
EXCEL_DB_PATH = str(resolve_excel_path(EXCEL_DB_PATH))

def _map_main_source_to_type_name(main_source: str) -> str:
    s = (main_source or "").strip().lower()
    if "protein" in s:
        return "protein"
    if "carb" in s:
        return "carbohydrate"
    if "fat" in s or "oil" in s:
        return "fat"
    return s or ""

def _to_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
    except Exception:
        if x is None or (isinstance(x, float) and (x != x)):
            return default
    try:
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return default
            x = x.replace(",", ".")
        return float(x)
    except Exception:
        return default

def _to_int(x, default=0):
    f = _to_float(x, None)
    if f is None:
        return default
    try:
        return int(round(f))
    except Exception:
        return default

def _to_str(x, default=""):
    try:
        if x is None or pd.isna(x):
            return default
    except Exception:
        if x is None or (isinstance(x, float) and (x != x)):
            return default
    s = str(x).strip()
    return s if s else default

def load_foods_from_excel(xlsx_path: str) -> Dict[str, Food]:
    xls = pd.ExcelFile(xlsx_path)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        lower = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    col_type        = col("Type")
    col_grams       = col("Grams")
    col_cal         = col("Calories", "Kcal", "Energy")
    col_carbs       = col("Carbohydrates", "Carbs")
    col_prot        = col("Proteins", "Protein")
    col_fat         = col("Fats", "Fat")
    col_source      = col("Main Source", "Source", "Group")
    col_order       = col("Order", "Sort", "Rank")
    col_subcat      = col("Subcategory", "Sub-category", "Sub cat")
    col_tags        = col("AttributeTags", "Tags", "Attributes")
    col_slang       = col("Egyptian Slang", "Slang")
    col_arabic      = col("Arabic")

    foods: Dict[str, Food] = {}

    for _, row in df.iterrows():
        name = _to_str(row.get(col_type))
        if not name:
            continue

        grams_base = _to_float(row.get(col_grams), 100.0)
        scale = 100.0 / grams_base if grams_base > 0 else 1.0

        cals  = _to_float(row.get(col_cal), 0.0)  * scale
        carbs = _to_float(row.get(col_carbs), 0.0) * scale
        prot  = _to_float(row.get(col_prot), 0.0)  * scale
        fat   = _to_float(row.get(col_fat), 0.0)   * scale

        type_name = _map_main_source_to_type_name(_to_str(row.get(col_source)))
        subcat    = _to_str(row.get(col_subcat), None) or None
        order     = _to_int(row.get(col_order), 0)

        raw_tags = _to_str(row.get(col_tags))
        tags = ",".join(t.strip() for t in raw_tags.split(",") if t.strip()) if raw_tags else ""

        f = Food(
            name=name,
            type_name=type_name,
            sub_category=subcat,
            calories=cals,
            protein=prot,
            carbohydrates=carbs,
            fat=fat,
            tags=tags,
            order=order
        )

        foods[name.lower()] = f

        slang  = _to_str(row.get(col_slang))
        arabic = _to_str(row.get(col_arabic))
        if slang and slang.lower() not in foods:
            foods[slang.lower()] = f
        if arabic and arabic.lower() not in foods:
            foods[arabic.lower()] = f

    return foods

NUTRITION_DB: Dict[str, Food] = load_foods_from_excel(EXCEL_DB_PATH)


# ========================= Tag helpers =========================

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
        a1 = (getattr(alt,   sec[0]) / 100.0) * alt_qty_g
        a2 = (getattr(alt,   sec[1]) / 100.0) * alt_qty_g

        def close_enough(target, actual):
            abs_diff = abs(actual - target)
            tol_abs  = 8.0
            tol_rel  = 0.25 * max(1.0, target)
            return abs_diff <= max(tol_abs, tol_rel)

        if close_enough(t1, a1) and close_enough(t2, a2):
            return "Good Match"
        return "High Variance"

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


# ========================= Constants & UI Labels =========================

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


# ========================= Preference keywords & helpers =========================

KEYS = {
    "meat_red": ["beef", "kofta", "lamb", "steak", "mince", "shawarma"],
    "poultry": ["chicken", "turkey", "breast", "thigh", "wings"],
    "seafood": ["fish", "tuna", "salmon", "shrimp", "prawn"],
    "pork": ["pork", "bacon", "ham"],
    "eggs": ["egg", "eggs"],
    "legumes": ["lentil", "lentils", "chickpea", "chickpeas", "beans", "bean", "ful", "fava"],
    "soy": ["soy", "tofu", "edamame", "tempeh"],
    "nuts": ["nuts","mixed nuts","almond","almonds","almond butter","peanut","peanuts","peanut butter","hazelnut","hazelnuts","walnut","walnuts","cashew","cashews","pistachio","pistachios","pecan","pecans","macadamia","macadamias","brazil nut","brazil nuts","pine nut","pine nuts","pignoli","tahini","tahina","sesame","sesame seeds"],
    "crucifer": ["broccoli", "cauliflower", "cabbage"],
    "nightshade": ["tomato", "pepper", "eggplant"],
    "starchy": ["potato", "potatoes", "corn"],
    "sweets": ["honey", "dessert", "sweet", "candy", "sugar"],
    "spicy": ["shakshuka", "harissa", "spicy", "chili", "chilli", "hot sauce"],
    "dairy": ["yogurt", "yoghurt", "labneh", "cheese", "milk", "cottage cheese"],
    "gluten": ["bread", "toast", "pasta", "bulgur", "freekeh", "wheat", "barley"],
}

ANIMAL_DERIV_KEYWORDS = [
    "basterma","pastrami","salami","pepperoni","anchovy","anchovies","fish sauce",
    "lard","collagen","gelatin","bovine gelatin"
]


def contains_any(text: str, words: List[str]) -> bool:
    t = text.lower()
    for w in words:
        pattern = r'\b' + re.escape(w.lower()) + r's?\b'
        if re.search(pattern, t):
            return True
    return False


# ========================= Constraint checks =========================

def option_violates_prefs(opt: Dict[str, Any], allergies: List[str], dislikes: List[str]) -> bool:
    """
    HARD filter only for non-negotiable diet rules (vegan/vegetarian, dairy-free, gluten-free).
    Do not filter on 'dislikes' here (swapper handles those).
    """
    text = (opt.get("option_name","") or "").lower()

    def has_any(words: List[str]) -> bool:
        return contains_any(text, words)

    if "Vegan" in allergies:
        if (
            has_any(KEYS["meat_red"] + KEYS["poultry"] + KEYS["seafood"] + KEYS["pork"] + KEYS["eggs"] + KEYS["dairy"])
            or any(w in text for w in ANIMAL_DERIV_KEYWORDS)
        ):
            return True
    elif "Vegetarian" in allergies:
        if has_any(KEYS["meat_red"] + KEYS["poultry"] + KEYS["seafood"] + KEYS["pork"]) or any(w in text for w in ["gelatin","collagen"]):
            return True

    if "Dairy-free" in allergies and has_any(KEYS["dairy"]):
        return True
    if "Gluten-free" in allergies and has_any(KEYS["gluten"]):
        return True

    return False


# ========================= DB-driven replacement & helpers =========================

def _allowed_food_by_constraints(food: Food, allergies: List[str], dislikes: List[str]) -> bool:
    name = (food.name or "").lower()
    tags = (food.tags or "").lower()

    if "Vegan" in allergies:
        if (
            food_has_any_tag(food, ["meat_product","fish_seafood","dairy_product","egg"])
            or contains_any(name, KEYS["meat_red"] + KEYS["poultry"] + KEYS["seafood"] + KEYS["pork"] + KEYS["eggs"] + KEYS["dairy"])
            or any(w in name for w in ANIMAL_DERIV_KEYWORDS)
        ):
            return False

    if "Vegetarian" in allergies:
        if (
            food_has_any_tag(food, ["meat_product","fish_seafood"])
            or contains_any(name, KEYS["meat_red"] + KEYS["poultry"] + KEYS["seafood"] + KEYS["pork"])
            or any(w in name for w in ["gelatin","collagen"])
        ):
            return False

    if "Dairy-free" in allergies:
        if (food_has_any_tag(food, ["dairy_product","yogurt","milk","cheese"]) or contains_any(name, KEYS["dairy"])):
            return False

    if "Gluten-free" in allergies:
        if (food_has_any_tag(food, ["gluten","wheat","barley","pasta","bread"]) or contains_any(name, KEYS["gluten"])):
            return False

    if "Seafood (fish & shellfish)" in dislikes:
        if food_has_any_tag(food, ["fish_seafood"]) or contains_any(name, KEYS["seafood"]):
            return False

    if "Red meat (beef, lamb, etc.)" in dislikes:
        if contains_any(name, KEYS["meat_red"]):
            return False
        if food_has_any_tag(food, ["meat_product"]) and not contains_any(name, KEYS["poultry"] + KEYS["pork"]):
            return False

    if "Poultry (chicken, turkey)" in dislikes and contains_any(name, KEYS["poultry"]):
        return False

    if "Pork" in dislikes and (contains_any(name, KEYS["pork"]) or "pork" in tags):
        return False

    if "Eggs" in dislikes and ("egg" in tags or contains_any(name, KEYS["eggs"])):
        return False

    if "Legumes (beans, lentils, peas)" in dislikes and (food_has_any_tag(food, ["legume_bean_pea"]) or contains_any(name, KEYS["legumes"])):
        return False

    if "Soy products" in dislikes and ("soy" in tags or contains_any(name, KEYS["soy"])):
        return False

    if "Nuts & seeds" in dislikes and (food_has_any_tag(food, ["nut_based","seed_based"]) or contains_any(name, KEYS["nuts"])):
        return False

    if "Cruciferous veg (broccoli, cauliflower, cabbage)" in dislikes and contains_any(name, KEYS["crucifer"]):
        return False

    if "Nightshades (tomato, pepper, eggplant)" in dislikes and contains_any(name, KEYS["nightshade"]):
        return False

    if "Starchy veg (potato, corn)" in dislikes and contains_any(name, KEYS["starchy"]):
        return False

    if "Sweets & desserts" in dislikes and contains_any(name, KEYS["sweets"]):
        return False

    if "Spicy foods" in dislikes and contains_any(name, KEYS["spicy"]):
        return False

    return True


def _find_original_foods_in_text(option_text: str) -> List[Tuple[str, Food]]:
    text = option_text.lower()
    hits: List[Tuple[str, Food]] = []
    seen_pairs = set()
    for key, food in NUTRITION_DB.items():
        key = key.strip().lower()
        if not key:
            continue
        pat = r"\b" + re.escape(key) + r"s?\b"
        m = re.search(pat, text)
        if m:
            matched = m.group(0)
            pair = (matched, food)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                hits.append(pair)
    return hits

def _is_powder_or_supplement(food: Food) -> bool:
    name = (food.name or "")
    if food_has_any_tag(food, ["powdered"]):
        return True
    # include yeast/collagen/gelatin, isolates, concentrates, peptides
    return bool(re.search(r"\b(powder|isolate|concentrate|collagen|gelatin|peptides|yeast)\b", name, re.I))

def _is_whole_food_protein(food: Food) -> bool:
    name = (food.name or "").lower()
    return (
        "tofu" in name or "tempeh" in name or "seitan" in name
        or food_has_any_tag(food, ["legume_bean_pea","nut_based","seed_based"])
    )

def _candidate_alternatives_for(original: Food, allergies: List[str], dislikes: List[str]) -> List[Food]:
    """
    Prefer same sub_category/type/functional tags.
    Rank whole-food proteins first; supplements last.
    """
    orig_tags = set(t.lower() for t in (original.tags or "").split(",") if t.strip())
    candidates: List[Food] = []
    unique_foods = list(set(NUTRITION_DB.values()))

    for f in unique_foods:
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

    if not candidates:
        candidates = [
            f for f in unique_foods
            if f.name != original.name
            and _allowed_food_by_constraints(f, allergies, dislikes)
            and (original.type_name and f.type_name and f.type_name.lower() == original.type_name.lower())
        ]

    def rank_key(f: Food):
        powder_penalty = 2 if _is_powder_or_supplement(f) else 0
        whole_bonus = -1 if _is_whole_food_protein(f) else 0
        return (powder_penalty + whole_bonus, f.order or 9999, f.name.lower())

    candidates.sort(key=rank_key)
    return candidates


def _best_swap_for(original: Food,
                   allergies: List[str],
                   dislikes: List[str],
                   assumed_original_qty_g: int = 150,
                   slot_type: Optional[str] = None,
                   ) -> Optional[Tuple[Food, Dict[str, Any]]]:
    cands = _candidate_alternatives_for(original, allergies, dislikes)
    if not cands:
        return None

    tier_index = {"Good Match": 0, "High Variance": 1, "Unrecommended": 2}
    scored: List[Tuple[int, int, Food, Dict[str, Any]]] = []

    for alt in cands:
        result = SCORER.compute_swap(original, alt, assumed_original_qty_g)
        if not result.get("ok"):
            continue
        score = tier_index[result["suitability"]]
        if slot_type == "breakfast" and food_has_any_tag(alt, ["meat_product", "fish_seafood"]):
            score += 1
        scored.append((score, alt.order or 0, alt, result))

    if not scored:
        return None

    best_tier = min(s for s, _, _, _ in scored)

    by_tier: Dict[int, List[Tuple[int, Food, Dict[str, Any]]]] = {}
    for s, ordr, a, r in scored:
        by_tier.setdefault(s, []).append((ordr, a, r))
    for t in by_tier:
        by_tier[t].sort(key=lambda x: x[0])

    pool: List[Tuple[int, Food, Dict[str, Any]]] = []
    TOP_K = 8
    for t in (best_tier, best_tier + 1):
        if t in by_tier:
            for item in by_tier[t]:
                pool.append(item)
                if len(pool) >= TOP_K:
                    break
        if len(pool) >= TOP_K:
            break

        if not pool:
            return None

    # deterministically pick the top-ordered alternative in the best tier
    ordr, alt, swap = by_tier[best_tier][0]
    return alt, swap



def _replace_food_and_qty_once(text: str, token: str, alt_name: str, alt_qty_g: float) -> str:
    qty = int(round(alt_qty_g))
    adj = r'(?:lean|smoked|minced|ground|boneless|skinless|cooked|raw|grilled|baked|roasted|boiled|steamed)'
    pre_qty    = r'(?:\b\d{1,4}\s*(?:g|grams)\b|\b\d{1,4}\b)\s*'
    pre_part   = rf'(?:{pre_qty}(?:{adj}\s*){{0,3}})?'
    token_pat  = r'\b' + re.escape(token) + r's?\b'
    post_grams = r'(?:\s*\b\d{1,4}\s*(?:g|grams)\b)?'
    pattern = rf'(?i){pre_part}({token_pat}){post_grams}'
    repl = f'{qty}g {alt_name}' if alt_name else ''
    out = re.sub(pattern, repl, text, count=1)
    out = re.sub(r'\s{2,}', ' ', out)
    out = re.sub(r'\s*\+\s*', ' + ', out)
    out = re.sub(r'(^\s*\+\s*|\s*\+\s*$)', '', out).strip(' .')
    return out

def _cleanup_artifacts(text: str) -> str:
    text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\+\s*", " + ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" .")
    return text

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

def _normalize_quantity_first(text: str) -> str:
    def fix_item(s: str) -> str:
        s = s.strip(" .")
        m = re.match(r'(?i)^\s*(\d{1,4})\s*(g|grams)\b\s+(.+?)\s*$', s)
        if m:
            n, _, name = m.groups()
            return f'{n}g {name}'
        m = re.match(r'(?i)^(.+?)\s+(\d{1,4})\s*(g|grams)\b\s*$', s)
        if m:
            name, n, _ = m.groups()
            return f'{n}g {name}'
        s = re.sub(r'(?i)\b(\d{1,4})\s*(g|grams)\b', lambda k: f"{k.group(1)}g", s)
        return s
    parts = [p for p in re.split(r'\s*\+\s*', text) if p.strip(" .")]
    parts = [fix_item(p) for p in parts]
    return " + ".join(parts)

def _fix_orphan_leading_counts(text: str) -> str:
    segs = [s.strip() for s in re.split(r'\s*\+\s*', text) if s.strip()]
    fixed = []
    for s in segs:
        s = re.sub(r'(?i)^\s*(\d{1,4})\s+(?=\d{1,4}\s*g\b)', '', s)
        fixed.append(s)
    return ' + '.join(fixed)

def _pick_vegan_protein_replacement(allergies: List[str], dislikes: List[str]) -> str:
    """
    Choose a sensible vegan protein replacement for eggs, honoring constraints.
    Preference order tries soy first unless user dislikes soy; then seitan; then legumes.
    """
    prefer_no_soy = "Soy products" in dislikes
    candidates_order = (
        ["Seitan", "Chickpeas", "Lentils", "Fava Beans (Ful)"]
        if prefer_no_soy
        else ["Tofu", "Tofu Skin (Yuba)", "Tempeh", "Seitan", "Chickpeas", "Lentils", "Fava Beans (Ful)"]
    )

    for name in candidates_order:
        f = NUTRITION_DB.get(name.lower())
        if f and _allowed_food_by_constraints(f, allergies, dislikes):
            return f.name

    for name in candidates_order:
        if prefer_no_soy and any(s in name.lower() for s in ["tofu", "tempeh"]):
            continue
        return name

    return "Tofu"


def _swap_egg_phrases_to_vegan(text: str, allergies: List[str], dislikes: List[str]) -> str:
    """
    Replace egg phrases like:
      - '3 whole', '2 whole eggs', '3 whites', 'egg whites', 'eggs', 'egg yolks'
    with a vegan protein (e.g., '150g Tofu'), scaling by count when available.
    """
    t = text or ""
    need_swap = ("Vegan" in allergies) or ("Eggs" in dislikes)
    if not need_swap:
        return t

    repl_name = _pick_vegan_protein_replacement(allergies, dislikes)

    patterns = [
        r'(?i)\b(\d{1,3})\s*(?:whole|whites?|yolks?)\s*(?:eggs?)?\b',
        r'(?i)\b(\d{1,3})\s*eggs?\b',
        r'(?i)\begg whites?\b',
        r'(?i)\begg yolks?\b',
        r'(?i)\beggs?\b',
    ]

    def grams_for_match(m: re.Match) -> int:
        if m and m.lastindex:
            try:
                n = int(m.group(1))
                return max(100, min(250, n * 75))
            except Exception:
                pass
        return 150

    for pat in patterns:
        def _do(m):
            grams = grams_for_match(m)
            return f"{grams}g {repl_name}"
        t = re.sub(pat, _do, t)

    t = re.sub(r'(?i)\b(?:whole|whites?|yolks?)\b', '', t)
    t = re.sub(r'\s{2,}', ' ', t)
    t = re.sub(r'\s*\+\s*', ' + ', t)
    t = re.sub(r'(^\s*\+\s*|\s*\+\s*$)', '', t).strip(' .')

    return t

def _remove_redundant_paren_units(text: str) -> str:
    # Remove trailing parenthetical unit labels that are redundant in display
    # e.g., "1 cup Milk (cup)" → "1 cup Milk"
    return re.sub(r'\s*\((?:cup|cups|ml|milliliters?|l|liters?|tbsp|tablespoons?|tsp|teaspoons?)\)', '', text, flags=re.I)

def _clean_food_name_units(name: str) -> str:
    # When inserting a DB Food.name, strip any unit-like parenthetical suffixes
    return re.sub(r'\s*\((?:cup|cups|ml|milliliters?|l|liters?|tbsp|tablespoons?|tsp|teaspoons?)\)\s*$', '', name, flags=re.I).strip()


def _fix_double_quantities(text: str) -> str:
    adj = r'(?:lean|smoked|minced|ground|boneless|skinless|cooked|raw|grilled|baked|roasted|boiled|steamed)'
    pat = rf'(?i)\b\d{{1,4}}\s*g\s*(?:{adj}\s*){{0,3}}(\d{{1,4}})\s*g\s+([A-Za-z][^+]*?)\b'
    def repl(m):
        qty2 = m.group(1)
        name = m.group(2).strip()
        return f'{qty2}g {name}'
    before = None
    cur = text
    for _ in range(3):
        cur = re.sub(pat, repl, cur)
        if cur == before:
            break
        before = cur
    return cur

def _parse_option_items_freeform(option_text: str) -> List[Dict[str, Any]]:
    parts = [p.strip(" .") for p in option_text.split("+")]
    return [{"food": p, "grams": None} for p in parts if p]

# ----------------------- NEW: Calorie estimation & rebalancing -----------------------

def _lookup_food_by_name_approx(name: str) -> Optional[Food]:
    """Loose lookup: try exact, lower, strip plurals, and collapse spaces."""
    key = name.strip().lower()
    if key in NUTRITION_DB:
        return NUTRITION_DB[key]
    # drop trailing s
    key2 = re.sub(r's\b', '', key)
    if key2 in NUTRITION_DB:
        return NUTRITION_DB[key2]
    # collapse multiple spaces
    key3 = re.sub(r'\s+', ' ', key2).strip()
    return NUTRITION_DB.get(key3)

def _find_grammed_items(text: str) -> List[Dict[str, Any]]:
    """
    Return items like [{'span':(start,end), 'qty':120, 'name':'Tofu', 'food': Food}, ...]
    Only items written as '<grams>g <Food Name>'.
    """
    items = []
    for m in re.finditer(r'(?i)\b(\d{1,4})\s*g\s+([A-Za-z][^+]+?)(?=$|\s*\+)', text):
        qty = int(m.group(1))
        raw_name = m.group(2).strip(" .")
        # strip trailing unit words in name fragments
        name = re.sub(r'\s*\b(grams?|g)\b', '', raw_name, flags=re.I).strip()
        food = _lookup_food_by_name_approx(name)
        items.append({"span": (m.start(), m.end()), "qty": qty, "name": name, "food": food})
    return items

def _estimate_option_kcal(text: str) -> float:
    """
    Sum kcal for any '<grams>g <Food Name>' segments where Food exists in DB.
    Items without grams are ignored (safer than guessing).
    """
    total = 0.0
    for it in _find_grammed_items(text):
        f = it["food"]
        if f:
            total += (f.calories / 100.0) * it["qty"]
    return round(total, 1)

def _rebalance_option_kcal_to_match(text: str, kcal_target: float, tolerance: float = 50.0) -> str:
    """
    If current kcal deviates > tolerance, scale the largest kcal-contributing
    grammed item so the whole option lands within ±tolerance.
    """
    def round5(x):  # nicer grams
        return int(round(x / 5.0) * 5)

    for _ in range(2):  # at most two gentle nudges
        k_now = _estimate_option_kcal(text)
        if k_now == 0 or abs(k_now - kcal_target) <= tolerance:
            return text

        items = _find_grammed_items(text)
        items = [it for it in items if it["food"] is not None]
        if not items:
            return text

        # Pick the biggest kcal contributor
        items.sort(key=lambda it: (it["food"].calories / 100.0) * it["qty"], reverse=True)
        main = items[0]
        f = main["food"]
        if f.calories <= 0:
            return text

        # Compute new grams for that item
        # k_now = k_rest + (f.cal/100)*qty  -> qty_new = (kcal_target - k_rest) / (f.cal/100)
        k_rest = k_now - (f.calories / 100.0) * main["qty"]
        qty_new = (kcal_target - k_rest) / (f.calories / 100.0)
        qty_new = max(50, min(400, qty_new))  # clamp
        qty_new = round5(qty_new)

        # Replace in text
        start, end = main["span"]
        repl = f"{qty_new}g {main['name']}"
        text = text[:start] + repl + text[end:]
        # loop once more to check tolerance
    return text

# ------------------------------------------------------------------------------------
def _pick_safe_fat_replacement(allergies: List[str], dislikes: List[str]) -> Optional[Food]:
    """
    Pick a broadly allowed fat to replace nuts/seeds, strictly from DB.
    Uses Food.order (most-used first). Recognizes oils, butter/ghee, avocado, olives, etc.
    """
    uniq = set(NUTRITION_DB.values())

    def is_fat_like(f: Food) -> bool:
        name = (f.name or "").lower()
        # Primary signal: DB type says 'fat'
        if (f.type_name or "").lower() == "fat":
            return True
        # Tags that imply fat source
        if food_has_any_tag(f, ["oil", "fat", "butter", "ghee", "avocado", "olive"]):
            return True
        # Name heuristics to catch common whole-fat foods
        if any(w in name for w in [" oil", "oil ", "butter", "ghee", "tallow", "lard", "avocado", "olives", "olive"]):
            return True
        return False

    fats = [f for f in uniq if is_fat_like(f)]
    fats = [f for f in fats if _allowed_food_by_constraints(f, allergies, dislikes)]
    if not fats:
        return None

    # Most used first (lowest order), then lower fat (if tied), then name to stabilize
    fats.sort(key=lambda f: (f.order or 9999, f.fat, (f.name or "").lower()))
    return fats[0]


# ---------- DB-driven milk selection (no hard-coded names) ----------

_NUT_WORDS = [
    "almond","hazelnut","cashew","macadamia","pistachio","walnut",
    "peanut","pecan","brazil","pine","sesame","tahini","tahina"
]
_SOY_WORDS = ["soy","soya"]
_PLANT_WORDS = ["oat","rice","coconut","hemp","pea","flax","tigernut"]

def _is_milk_food(f: Food) -> bool:
    name = (f.name or "").lower()
    return ("milk" in name) or food_has_any_tag(f, ["milk"])

def _is_dairy_milk(f: Food) -> bool:
    name = (f.name or "").lower()
    # dairy if tagged as dairy_product or contains cow-related names (generic)
    return food_has_any_tag(f, ["dairy_product","milk"]) and not any(w in name for w in _PLANT_WORDS + _NUT_WORDS + _SOY_WORDS)

def _is_plant_milk(f: Food) -> bool:
    name = (f.name or "").lower()
    return (any(w in name for w in _PLANT_WORDS + _SOY_WORDS + _NUT_WORDS) or food_has_any_tag(f, ["vegan","plant_based"])) and "milk" in name

def _is_nut_or_seed_milk(f: Food) -> bool:
    name = (f.name or "").lower()
    return any(w in name for w in _NUT_WORDS) or food_has_any_tag(f, ["nut_based","seed_based"])

def _is_soy_milk(f: Food) -> bool:
    name = (f.name or "").lower()
    return ("milk" in name) and (any(w in name for w in _SOY_WORDS))

def _gather_milks_from_db() -> List[Food]:
    # unique foods (since NUTRITION_DB maps multiple keys to same Food)
    seen = set()
    out: List[Food] = []
    for f in set(NUTRITION_DB.values()):
        if id(f) in seen: 
            continue
        seen.add(id(f))
        if _is_milk_food(f):
            out.append(f)
    return out

def _choose_alt_milk_db(allergies: List[str], dislikes: List[str]) -> Optional[Food]:
    milks = _gather_milks_from_db()
    if not milks:
        return None

    dairy_free = "Dairy-free" in allergies
    soy_no = "Soy products" in dislikes
    nuts_no = "Nuts & seeds" in dislikes

    if dairy_free:
        # Only plant milks
        plant = [f for f in milks if _is_plant_milk(f)]
        if nuts_no:
            plant = [f for f in plant if not _is_nut_or_seed_milk(f)]
        if soy_no:
            plant = [f for f in plant if not _is_soy_milk(f)]
        if not plant:
            return None

        # Rank: most-used (lowest order) first, then preferred types (soy → oat → rice → coconut → pea → other),
        # then lower fat, then name for stability.
        def plant_rank(f: Food) -> Tuple[int, int, float, str]:
            name = (f.name or "").lower()
            if _is_soy_milk(f): pri = 0
            elif "oat" in name: pri = 1
            elif "rice" in name: pri = 2
            elif "coconut" in name: pri = 3
            elif "pea" in name: pri = 4
            else: pri = 5
            return ((f.order or 9999), pri, f.fat, name)

        plant.sort(key=plant_rank)
        return plant[0]

    # Not dairy-free: choose a dairy milk; prefer most-used (lowest order), then lower fat.
    dairy = [f for f in milks if _is_dairy_milk(f)]
    if dairy:
        dairy.sort(key=lambda f: ((f.order or 9999), f.fat, (f.name or "").lower()))
        return dairy[0]

    # Fallback: any non-nut plant milk (even if not dairy-free) to avoid nut milks.
    plant = [f for f in milks if _is_plant_milk(f)]
    if nuts_no:
        plant = [f for f in plant if not _is_nut_or_seed_milk(f)]
    plant.sort(key=lambda f: ((f.order or 9999), f.fat, (f.name or "").lower()))
    return plant[0] if plant else None

def _swap_nut_milks(text: str, allergies: List[str], dislikes: List[str]) -> str:
    # only act if nuts/seeds are disliked OR dairy-free requires plant milk replacement of nut milks
    if "Nuts & seeds" not in dislikes and "Dairy-free" not in allergies:
        return text

    alt_food = _choose_alt_milk_db(allergies, dislikes)  # strictly DB-driven (may be None)

    # Capture optional qty like "1 cup ", "250 ml ", etc.
    qty_pat = r'(?P<qty>\b\d{1,3}\s*(?:cups?\b|ml\b|milliliters?\b)\s*)?'
    nuts_pat = r'(almond|hazelnut|cashew|macadamia|pistachio|walnut|peanut|pecan|brazil\s+nut|brazil\s+nuts?|pine\s+nut|pine\s+nuts?|sesame|tahini|tahina)'
    pat = rf'(?i)\b{qty_pat}{nuts_pat}\s+milk\b'

    # If we have a DB alternative, use its canonical DB name
    if alt_food:
        def repl_db(m):
            qty = (m.group('qty') or '').strip()
            return (f"{qty} {_clean_food_name_units(alt_food.name)}").strip()
        out = re.sub(pat, repl_db, text)
        return re.sub(r'\s{2,}', ' ', out).strip()
    milks = _gather_milks_from_db()
    fallback = None
    if milks:
        # prefer a non-nut milk of any kind
        non_nut = [f for f in milks if not _is_nut_or_seed_milk(f)]
        if non_nut:
            non_nut.sort(key=lambda f: (f.fat, (f.name or "").lower()))
            fallback = non_nut[0]

    if fallback:
        def repl_fb(m):
            qty = (m.group('qty') or '').strip()
            return (f"{qty} {_clean_food_name_units(fallback.name)}").strip()

        out = re.sub(pat, repl_fb, text)
    else:
        # Last resort: strip the nut word; leave "milk" and qty intact
        def repl_strip(m):
            qty = (m.group('qty') or '').strip()
            return (f"{qty} milk").strip()
        out = re.sub(pat, repl_strip, text)

    return re.sub(r'\s{2,}', ' ', out).strip()


def _swap_disliked_generic_terms(text: str, allergies: List[str], dislikes: List[str]) -> str:
    if "Nuts & seeds" not in dislikes:
        return text

    text = _swap_nut_milks(text, allergies, dislikes)
    alt_fat = _pick_safe_fat_replacement(allergies, dislikes)
    if not alt_fat:
        # No allowed fat: delete nut phrases and tidy.
        patterns = [
            r'(?i)\+?\s*\d{1,3}\s*g\s*(?:mixed\s+)?nuts?\b',
            r'(?i)\+?\s*\d{1,3}\s*g\s*(almonds?|walnuts?|hazelnuts?|cashews?|pistachios?|pecans?|macadamias?|brazil\s+nuts?|pine\s+nuts?|sesame|tahini|tahina)\b',
            r'(?i)\+?\s*(almonds?|walnuts?|hazelnuts?|cashews?|pistachios?|pecans?|macadamias?|brazil\s+nuts?|pine\s+nuts?|sesame|tahini|tahina)\b'
        ]
        out = text
        for pat in patterns:
            out = re.sub(pat, '', out)
        return _cleanup_artifacts(_dedupe_plus_items(out))

    # Convert grams of nuts -> grams of oil ≈ 0.5x (nuts ~ 6kcal/g, oil ~ 9kcal/g)
    def repl_grammed(m):
        qty = int(m.group(1))
        oil_g = max(5, int(round(qty * 0.5)))
        return f"{oil_g}g {alt_fat.name}"

    out = text
    # 1) "<XX> g nuts"
    out = re.sub(r'(?i)\b(\d{1,3})\s*g\s*(?:mixed\s+)?nuts?\b', repl_grammed, out)

    # 2) "<XX> g <specific nut name>"  OR  "<specific nut name>"
    nut_names = r'(almonds?|walnuts?|hazelnuts?|cashews?|pistachios?|pecans?|macadamias?|brazil\s+nuts?|pine\s+nuts?|sesame|tahini|tahina)(?!\s*(milk|oil)\b)'
    def repl_specific(m):
        g = m.group('g')
        if g:
            qty = int(g)
            oil_g = max(5, int(round(qty * 0.5)))
        else:
            oil_g = 15
        return f"{oil_g}g {alt_fat.name}"

    out = re.sub(rf'(?i)\b(?:(?P<g>\d{{1,3}})\s*g\s*)?{nut_names}\b', repl_specific, out)

    # Tidy formatting
    out = _normalize_quantity_first(out)
    out = _cleanup_artifacts(_dedupe_plus_items(out))
    return out


def _enrich_powder_meal_if_needed(text: str) -> str:
    """
    If a meal is mostly supplements (powders/yeast/collagen/gelatin) and lacks real food,
    add simple carbs/fruit so it’s sensible.
    """
    t = (text or "").lower()
    has_supp = re.search(r"\b(powder|isolate|concentrate|collagen|gelatin|peptides|yeast)\b", t)
    has_real = re.search(r"\b(tofu|tempeh|seitan|bean|beans|chickpea|lentil|oats?|toast|rice|potato|banana|apple)\b", t)
    if has_supp and not has_real:
        return (text + " + 50g oats + 1 banana").strip(" .")
    return text

def rewrite_option_text_for_prefs(option_text: str,
                                  allergies: List[str],
                                  dislikes: List[str],
                                  slot_type: Optional[str] = None) -> str:
    # Keep original kcal snapshot (based on grammed items only)
    kcal_before = _estimate_option_kcal(option_text)

    new_text = option_text
    matches = _find_original_foods_in_text(new_text)

    for matched_token, original in matches:
        if _allowed_food_by_constraints(original, allergies, dislikes):
            continue

        best = _best_swap_for(
            original,
            allergies,
            dislikes,
            assumed_original_qty_g=150,
            slot_type=slot_type
        )
        if best is not None:
            alt, swap = best
            new_text = _replace_food_and_qty_once(
                new_text,
                token=matched_token,
                alt_name=alt.name,
                alt_qty_g=swap['quantity_g']
            )
        else:
            new_text = _replace_food_and_qty_once(
                new_text,
                token=matched_token,
                alt_name="",
                alt_qty_g=0
            )

    new_text = _cleanup_artifacts(new_text)
    new_text = _dedupe_plus_items(new_text)
    new_text = _normalize_quantity_first(new_text)
    new_text = _fix_double_quantities(new_text)
    new_text = _fix_orphan_leading_counts(new_text)
    new_text = _swap_egg_phrases_to_vegan(new_text, allergies, dislikes)
    new_text = _swap_disliked_generic_terms(new_text, allergies, dislikes)
    new_text = _enrich_powder_meal_if_needed(new_text)
    kcal_after = _estimate_option_kcal(new_text)
    if kcal_before > 0 and kcal_after > 0 and abs(kcal_after - kcal_before) > 50.0:
        new_text = _rebalance_option_kcal_to_match(new_text, kcal_target=kcal_before, tolerance=50.0)
        # final tidy
        new_text = _cleanup_artifacts(new_text)

    new_text = _remove_redundant_paren_units(new_text)

    return new_text



# ========================= Apply replacements across a diet =========================

def apply_replacements_to_diet_options(diet: Dict[str, Any],
                                       allergies: List[str],
                                       dislikes: List[str]) -> Dict[str, Any]:
    new_d = dict(diet)
    new_slots: Dict[str, List[Dict[str, Any]]] = {}
    for slot, opts in diet["slots"].items():
        slot_type = "breakfast" if str(slot).strip().lower().startswith("meal 1") else None
        new_opts = []
        for opt in opts:
            txt = opt.get("option_name", "")
            fixed = rewrite_option_text_for_prefs(
                txt, allergies, dislikes, slot_type=slot_type
            )
            new_opt = dict(opt)
            new_opt["option_name"] = fixed
            new_opt["items"] = _parse_option_items_freeform(fixed)
            new_opts.append(new_opt)
        new_slots[slot] = new_opts
    new_d["slots"] = new_slots
    return new_d


# ========================= JSON band loader (unchanged structure) =========================

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
    avoid_names = [n for n in (avoid_names or []) if n]
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

    if rng is None:
        rng = random.Random()

    base_seq = sequence_all[:]
    sequence = base_seq if desired_slots is None else base_seq[:desired_slots]

    chosen: Dict[str, Dict[str, Any]] = OrderedDict()
    prev_slot_opts: Optional[List[Dict[str, Any]]] = None
    prev_slot_pick_name: Optional[str] = None

    def _pick_from_slot(slot: str) -> Optional[Dict[str, Any]]:
        nonlocal prev_slot_opts, prev_slot_pick_name
        opts = diet["slots"].get(slot, [])
        if not opts:
            return None

        avoid_names: List[str] = []
        # avoid repeating the exact same option earlier this day
        avoid_names.extend([p.get("option_name","") for p in chosen.values() if p.get("option_name")])
        avoid_prob = 0.95 if avoid_names else 0.0

        if _same_option_set(prev_slot_opts, opts) and prev_slot_pick_name:
            avoid_names.append(prev_slot_pick_name)
            avoid_prob = max(avoid_prob, 0.60)
        if previous_day_choices:
            y_choice = previous_day_choices.get(slot)
            if y_choice:
                avoid_names.append(y_choice)
                avoid_prob = max(avoid_prob, 0.60)

        idx = _pick_option_index(rng, opts, avoid_names=avoid_names, avoid_prob=avoid_prob)
        pick = opts[idx]
        prev_slot_opts = opts
        prev_slot_pick_name = pick.get("option_name")
        return pick

    for slot in sequence:
        pick = _pick_from_slot(slot)
        if pick:
            chosen[slot] = pick

    while desired_slots is not None and len(chosen) < desired_slots:
        for slot in base_seq:
            if len(chosen) >= desired_slots:
                break
            pick = _pick_from_slot(slot)
            if pick:
                key = slot
                suffix = 2
                while key in chosen:
                    key = f"{slot} ({suffix})"
                    suffix += 1
                chosen[key] = pick
        if not base_seq:
            break

    if not chosen:
        raise ValueError(f"No schedulable options found for requested slots in diet '{diet.get('diet_label','')}'.")

    chosen = _renumber_slots_meals_first(chosen)

    return {
        "diet_label": diet.get("diet_label"),
        "diet_kcal": diet.get("diet_kcal"),
        "slots": chosen
    }


def _renumber_slots_meals_first(chosen: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = OrderedDict()
    meal_i, snack_i = 1, 1
    for slot, pick in chosen.items():
        if _slot_is_meal(slot):
            key = f"meal {meal_i}"; meal_i += 1
        else:
            key = "snack" if snack_i == 1 else f"snack {snack_i}"
            snack_i += 1
        out[key] = pick
    return out


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
    last_choice_per_slot: Dict[str, str] = {}

    for d in range(days):
        base_diet = diets[d % len(diets)]
        swapped_diet = apply_replacements_to_diet_options(base_diet, allergies, dislikes)
        safe_diet = filter_diet_options_by_prefs(swapped_diet, allergies, dislikes)

        if not _nonempty_slots(safe_diet):
            safe_diet = swapped_diet
        if not _nonempty_slots(safe_diet):
            safe_diet = base_diet

        day_plan = build_one_day_from_diet(
            safe_diet,
            d,
            desired_slots=meals_per_day,
            rng=rng,
            previous_day_choices=last_choice_per_slot
        )

        # Keep a light post-pass in case of rare duplicates
        seen_names = set()
        for slot, opt in list(day_plan["slots"].items()):
            name = opt.get("option_name", "")
            if name in seen_names:
                opts = safe_diet["slots"].get(slot, [])
                alts = [o for o in opts if o.get("option_name") not in seen_names]
                if alts:
                    day_plan["slots"][slot] = rng.choice(alts)
                    name = day_plan["slots"][slot].get("option_name","")
            seen_names.add(name)

        plan.append(day_plan)

        for slot, opt in day_plan["slots"].items():
            last_choice_per_slot[slot] = opt.get("option_name")

    return plan


# ========================= Export helpers (JSON outputs) =========================

def _items_from_option_name(option_name: str) -> List[str]:
    parts = [p.strip(" .") for p in re.split(r'\s*\+\s*', option_name) if p.strip(" .")]
    return parts

def _label_range_str(low: Optional[int], high: Optional[int]) -> str:
    if low and high:
        return f"{low}–{high}"
    if low and not high:
        return f"{low}+"
    return ""

def _structures_from_internal_diet(internal: Dict[str, Any]) -> List[Dict[str, Any]]:
    structure = internal.get("structure") or {}
    meals_per_day = structure.get("total_slots") or structure.get("meals") or len(_nonempty_slots(internal))
    meals: List[Dict[str, Any]] = []

    counters: Dict[str, int] = {"meal": 0, "snack": 0}

    for slot in internal.get("slot_sequence", []):
        opts = internal["slots"].get(slot, [])
        if not opts:
            continue

        if slot.lower().startswith("snack"):
            meal_number: Any = "Snack"
            counters["snack"] += 1
            if counters["snack"] > 1:
                meal_number = "Snack"
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
    payload = {"diets": []}
    for d in diets:
        item = {
            "label_range_kcal": label_range_kcal,
            "diet_label": d.get("diet_label") or "",
            "total_kcal": d.get("diet_kcal"),
            "macros": {},
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


# ========================= Preference filtering =========================

def filter_diet_options_by_prefs(diet: Dict[str, Any], allergies: List[str], dislikes: List[str]) -> Dict[str, Any]:
    cleaned_slots = {}
    for slot, opts in diet["slots"].items():
        kept = [o for o in opts if not option_violates_prefs(o, allergies, dislikes)]
        cleaned_slots[slot] = kept
    new_d = dict(diet)
    new_d["slots"] = cleaned_slots
    return new_d


# ========================= Interactive runner (JSON-first) =========================

def run_planner(
    json_dir: str = "JSON",
    weeks: int = 4,
    out_json_plan: str = "4_Week_Meal_Plan.json",
    user_profile_json: str = "user_profile.json",
    seed: Optional[int] = None
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

    mpd_idx = choose_from_list(["3 meals/day", "4 meals/day", "5 meals/day", "6 meals/day"],
                               "How many meals per day do you prefer?")
    meals_per_day = [3, 4, 5, 6][mpd_idx]
    print(f"Meals per day selected: {meals_per_day}")

    allergies, dislikes = collect_constraints_unified()

    bmr_val = bmr(gender, weight_kg, height_cm, age)
    tdee_val = TDEE(bmr_val, activity_factor)
    target_kcal_val, _ = target_kcal(tdee_val, goal)
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
        seed=rng_seed
    )

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
