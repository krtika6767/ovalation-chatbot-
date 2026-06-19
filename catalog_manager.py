import pandas as pd
import os
from thefuzz import fuzz

CATALOG_FOLDER = "catalogs"
all_products = []

# =====================================
# LOAD CATALOGS
# =====================================
def load_catalogs():
    global all_products
    all_products = []

    if not os.path.exists(CATALOG_FOLDER):
        print("Catalog folder not found")
        return

    for file in os.listdir(CATALOG_FOLDER):
        if file.endswith((".xlsx", ".csv")):
            path = os.path.join(CATALOG_FOLDER, file)
            try:
                if file.endswith(".csv"):
                    df = pd.read_csv(path, dtype=str)
                else:
                    df = pd.read_excel(path, dtype=str)

                df.columns = [str(col).strip() for col in df.columns]
                df = df.fillna("")

                all_products.extend(df.to_dict("records"))
                print(f"Loaded {file}: {len(df)} products")

            except Exception as e:
                print(f"Error loading {file}: {e}")

    print(f"Total products loaded: {len(all_products)}")


load_catalogs()


# =====================================
# HELPERS
# =====================================
def get_code(product):
    return str(product.get("Product Code", product.get("Product Code ", ""))).strip()

def get_name(product):
    return str(product.get("Product Name", product.get("Product Name ", ""))).strip()

def get_category(product):
    return str(product.get("Category", product.get("Category ", ""))).strip()

def get_subcategory(product):
    for key in ["Sub Category", "SubCategory", "Sub-Category", "Subcategory", "subcategory"]:
        val = product.get(key, "")
        if val:
            return str(val).strip()
    return ""

def get_description(product):
    return str(product.get("Product Description", product.get("Product description", ""))).strip()

def is_valid_product(product):
    """
    A product is valid only if it has BOTH a non-empty product code AND a non-empty product name.
    This filters out rows where the code cell is blank (like the LANCE TIP issue).
    """
    code = get_code(product)
    name = get_name(product)
    return bool(code) and bool(name)


# =====================================
# EXACT PRODUCT LOOKUP BY CODE
# =====================================
def get_product_by_code(code):
    clean_code = str(code).strip().upper().replace(" ", "").replace("-", "")
    for product in all_products:
        if not is_valid_product(product):
            continue
        p_code = get_code(product)
        clean_p_code = p_code.upper().replace(" ", "").replace("-", "")
        if clean_p_code == clean_code and clean_p_code != "":
            return product
    return None


# =====================================
# FORMAT FULL PRODUCT DETAILS
# =====================================
def format_product_details(product):
    details = []
    code        = get_code(product)
    name        = get_name(product)
    category    = get_category(product)
    subcategory = get_subcategory(product)
    desc        = get_description(product)

    if code:        details.append(f"**Product Code:** {code}")
    if name:        details.append(f"**Product Name:** {name}")
    if category:    details.append(f"**Category:** {category}")
    if subcategory: details.append(f"**Sub-Category:** {subcategory}")
    if desc:        details.append(f"**Description:** {desc}")

    ignore_keys = {
        "Product Code", "Product Code ", "Product Name", "Product Name ",
        "Category", "Category ", "Sub Category", "SubCategory", "Sub-Category",
        "Subcategory", "subcategory", "Product Description", "Product description",
        "Image Name", "Type1", "Type2", "Type3", "Type4", "Type5", "Type6", "Type7"
    }
    for key, value in product.items():
        clean_key = key.strip()
        if clean_key not in ignore_keys and value:
            details.append(f"**{clean_key}:** {value}")

    return "\n".join(details)


# =====================================
# DETECT QUERY INTENT
# =====================================
def detect_intent(query):
    query_lower = query.lower().strip()

    category_triggers = [
        "number of products", "how many products", "list of products",
        "products under", "products in", "show all", "list all",
        "provide list", "provide me list", "provide the list"
    ]
    for trigger in category_triggers:
        if trigger in query_lower:
            return {"type": "category_list", "value": query_lower}

    for product in all_products:
        if not is_valid_product(product):
            continue
        code = get_code(product)
        code_lower   = code.lower()
        code_no_sep  = code_lower.replace("-", "").replace(" ", "")
        query_no_sep = query_lower.replace("-", "").replace(" ", "")
        if len(code_no_sep) > 2 and (code_lower in query_lower or code_no_sep in query_no_sep):
            return {"type": "exact_code", "value": code}

    for product in all_products:
        if not is_valid_product(product):
            continue
        name = get_name(product)
        if name and name.lower() in query_lower:
            return {"type": "product_name", "value": name}

    return {"type": "keyword_search", "value": query_lower}


# =====================================
# CATEGORY / SUBCATEGORY LISTING
# =====================================
def search_by_category(query_lower):
    matches = []
    seen_codes = set()

    for product in all_products:
        if not is_valid_product(product):
            continue
        category    = get_category(product).lower()
        subcategory = get_subcategory(product).lower()
        combined    = f"{category} {subcategory}"

        score = max(
            fuzz.partial_ratio(query_lower, category),
            fuzz.partial_ratio(query_lower, subcategory),
            fuzz.partial_ratio(query_lower, combined)
        )

        if score > 10:
            code = get_code(product)
            if code not in seen_codes:
                matches.append((score, product))
                seen_codes.add(code)

    matches.sort(key=lambda x: x[0], reverse=True)
    return matches


# =====================================
# KEYWORD SEARCH
# =====================================
def keyword_search(clean_query, max_results=20):
    stop_words = {
        "give", "me", "details", "about", "what", "is", "the", "price", "of",
        "can", "you", "show", "i", "want", "to", "know", "please", "tell", "a",
        "an", "for", "description", "describe", "info", "information", "product",
        "products", "all", "list", "find", "get"
    }
    words = clean_query.lower().split()
    important_words = [w for w in words if w not in stop_words]
    search_query = " ".join(important_words) if important_words else clean_query.lower()

    matches    = []
    seen_codes = set()

    for product in all_products:
        if not is_valid_product(product):   # ← skip blank-code rows
            continue

        code        = get_code(product)
        name        = get_name(product).lower()
        desc        = get_description(product).lower()
        category    = get_category(product).lower()
        subcategory = get_subcategory(product).lower()

        searchable = f"{name} {name} {name} {code.lower()} {subcategory} {category} {desc}"
        score      = fuzz.partial_token_set_ratio(search_query, searchable)

        if important_words and any(w in name for w in important_words):
            score = min(score + 15, 100)

        if important_words:
            name_hit = any(w in name or w in subcategory for w in important_words)
            if not name_hit:
                score -= 20

        if score > 65 and code not in seen_codes:
            matches.append((score, product))
            seen_codes.add(code)

    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[:max_results]


# =====================================
# MAIN SEARCH FUNCTION
# =====================================
def search_catalog(query, max_keyword_results=20):
    query       = str(query).strip()
    query_lower = query.lower()

    intent = detect_intent(query)

    if intent["type"] == "exact_code":
        product = get_product_by_code(intent["value"])
        if product:
            return f"Found exact product match:\n\n{format_product_details(product)}"

    if intent["type"] == "product_name":
        name_lower = intent["value"].lower()
        for product in all_products:
            if not is_valid_product(product):
                continue
            if get_name(product).lower() == name_lower:
                return f"Found product by name:\n\n{format_product_details(product)}"

    if intent["type"] == "category_list":
        cat_matches = search_by_category(query_lower)
        if cat_matches:
            best_category = get_category(cat_matches[0][1])
            result = f"Products under **{best_category}** ({len(cat_matches)} found):\n\n"
            for _, product in cat_matches:
                code = get_code(product)
                name = get_name(product)
                result += f"- **{code}**: {name}\n"
            return result

    matches = keyword_search(query_lower, max_results=max_keyword_results)

    if not matches:
        return "No products found matching your query."

    result = f"Found {len(matches)} matching product(s):\n\n"
    for i, (score, product) in enumerate(matches, 1):
        code   = get_code(product)
        name   = get_name(product)
        desc   = get_description(product)
        subcat = get_subcategory(product)
        desc_snippet = (desc[:200] + "...") if len(desc) > 200 else desc

        result += f"{i}. **{code}** — {name}\n"
        if subcat:
            result += f"   *Sub-category: {subcat}*\n"
        if desc_snippet:
            result += f"   {desc_snippet}\n"
        result += "\n"

    if len(matches) == max_keyword_results:
        result += f"*(Showing top {max_keyword_results} results.)*\n"

    return result