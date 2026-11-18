"""Clean Food.com and export chronological first-rating-day evaluation splits."""

import ast
import csv
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_START = "2010-04-25"
TEST_START = "2012-09-11"
COLUMNS = ("row_id", "user_id", "recipe_id", "date", "rating", "ingredients", "submitted")


def valid_date(value):
    try:
        return date.fromisoformat(value).isoformat() == value
    except (TypeError, ValueError):
        return False


def main():
    shared = ROOT / "data_prepared" / "shared"
    wyt = ROOT / "data_prepared" / "wyt"
    shared.mkdir(parents=True, exist_ok=True)
    wyt.mkdir(parents=True, exist_ok=True)
    recipes, seen_ids, recipe_counts = {}, set(), Counter()
    with (ROOT / "RAW_recipes.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            recipe_counts["raw"] += 1
            recipe_id = row["id"].strip()
            if not recipe_id:
                recipe_counts["missing_id"] += 1
                continue
            if recipe_id in seen_ids:
                raise ValueError(f"Duplicate recipe ID: {recipe_id}")
            seen_ids.add(recipe_id)
            if not valid_date(row["submitted"]):
                recipe_counts["invalid_submission_date"] += 1
                continue
            try:
                ingredients = ast.literal_eval(row["ingredients"])
            except (SyntaxError, ValueError):
                ingredients = None
            if not isinstance(ingredients, list) or not all(isinstance(i, str) for i in ingredients):
                recipe_counts["invalid_ingredient_list"] += 1
                continue
            ingredients = [re.sub(r"\s+", " ", i).strip().lower() for i in ingredients]
            ingredients = [i for i in ingredients if i]
            if not ingredients:
                recipe_counts["empty_ingredients"] += 1
                continue
            recipes[recipe_id] = (json.dumps(ingredients, ensure_ascii=False), row["submitted"])
            recipe_counts["retained"] += 1

    rows, first_day, cleaning, rating_counts = [], {}, Counter(), Counter()
    with (ROOT / "RAW_interactions.csv").open(encoding="utf-8-sig", newline="") as source:
        for row_id, row in enumerate(csv.DictReader(source), start=1):
            cleaning["raw"] += 1
            rating = row["rating"].strip()
            rating_counts[rating] += 1
            if rating == "0":
                cleaning["zero_rating"] += 1
                continue
            if rating not in {"1", "2", "3", "4", "5"}:
                cleaning["invalid_rating"] += 1
                continue
            user, recipe, day = row["user_id"].strip(), row["recipe_id"].strip(), row["date"]
            if not user or not recipe or not valid_date(day):
                cleaning["invalid_id_or_date"] += 1
                continue
            if recipe not in recipes:
                cleaning["recipe_missing_or_unusable"] += 1
                continue
            if day < recipes[recipe][1]:
                cleaning["rating_before_publication"] += 1
                continue
            rows.append((row_id, user, recipe, day, int(rating)))
            first_day[recipe] = min(first_day.get(recipe, day), day)
            cleaning["retained"] += 1

    assert cleaning["raw"] == sum(v for k, v in cleaning.items() if k != "raw")
    assert recipe_counts["raw"] == sum(v for k, v in recipe_counts.items() if k != "raw")
    rows.sort(key=lambda row: (row[3], row[0]))
    train = [r for r in rows if r[3] < VALIDATION_START]
    middle = [r for r in rows if VALIDATION_START <= r[3] < TEST_START]
    late = [r for r in rows if r[3] >= TEST_START]
    assert len(train) + len(middle) + len(late) == len(rows)
    train_users = Counter(r[1] for r in train)
    final_users = train_users + Counter(r[1] for r in middle)
    validation_ids = {r[0] for r in middle if r[1] in train_users and r[3] == first_day[r[2]]}
    validation = [r for r in middle if r[0] in validation_ids]
    test = [r for r in late if r[1] in final_users and r[3] == first_day[r[2]]]
    assert {r[2] for r in validation}.isdisjoint(r[2] for r in train)
    assert {r[2] for r in test}.isdisjoint(r[2] for r in train + middle)

    def write(path, header, values):
        with path.open("w", encoding="utf-8", newline="") as dest:
            writer = csv.writer(dest)
            writer.writerow(header)
            writer.writerows(values)

    def enrich(row):
        return (*row, *recipes[row[2]])

    write(wyt / "ratings_clean.csv", COLUMNS, (enrich(r) for r in rows))
    write(shared / "train.csv", COLUMNS, (enrich(r) for r in train))
    write(shared / "validation.csv", (*COLUMNS, "is_cold_start_eval"),
          ((*enrich(r), int(r[0] in validation_ids)) for r in middle))
    write(shared / "test_features.csv", (*COLUMNS[:4], *COLUMNS[5:]),
          ((*r[:4], *recipes[r[2]]) for r in test))
    write(wyt / "test_labels.csv", ("row_id", "rating"), ((r[0], r[4]) for r in test))

    def stats(records):
        return {"ratings": len(records), "users": len({r[1] for r in records}),
                "recipes": len({r[2] for r in records}),
                "first_date": records[0][3], "last_date": records[-1][3],
                "rating_distribution": dict(sorted(Counter(r[4] for r in records).items()))}

    report = {
        "validation_start": VALIDATION_START, "test_start": TEST_START,
        "recipe_cleaning": dict(recipe_counts), "interaction_cleaning": dict(cleaning),
        "raw_rating_distribution": dict(sorted(rating_counts.items())),
        "splits": {name: stats(records) for name, records in [
            ("all_clean", rows), ("train", train), ("validation_all", middle),
            ("validation_first_day", validation), ("test_all", late), ("test_first_day", test)]},
    }
    (ROOT / "data_prepared" / "data_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
