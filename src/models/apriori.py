import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations

# ----------------------------------------------------------------#
# Load data #
# ----------------------------------------------------------------#

df = pd.read_csv("../../data/processed/static_dataset3_discretized.csv", index_col=0)


transactions = df["Soil"].unique()
items = df["Temperature_width_disc"].unique()


def getTransDataset(df, transactions):
    df_result = pd.DataFrame(columns=["Transaction", "Items"])

    for trans in transactions:
        itemsList = []
        for index, row in df.iterrows():
            if trans in row.values:
                itemsList.append(row["Temperature_width_disc"])

        df_result = df_result.append(
            {"Transaction": trans, "Items": list(set(itemsList))}, ignore_index=True
        )

    return df_result


def generate_k_associations(filtered_dict, k):
    associations_list = []

    if all(isinstance(item, tuple) for item in filtered_dict):
        dict_list = list(filtered_dict.keys())
        comb = list(combinations(dict_list, k))
        associations_list = [tuple(set(sum(pair, ()))) for pair in comb]
    else:
        associations_list = list(combinations(filtered_dict, k))

    # Remove duplicates
    associations_list = list(set(associations_list))

    return associations_list


def calculateSupport(associations, df=None, previous_dict=None, k=None):
    item_counts = defaultdict(int)
    result_list = []

    if all(isinstance(item, tuple) for item in associations):
        if all(len(item) == 2 for item in associations):
            for items in df["Items"]:
                for ass in associations:
                    if ass in [(x, y) for x in items for y in items]:
                        item_counts[ass] += 1
        elif all(len(item) > 2 for item in associations):
            for items in associations:
                item_combinations = list(combinations(items, k))
                if all(comb in previous_dict for comb in item_combinations):
                    result_list.append(items)
        for itemset in result_list:
            support_count = sum(
                all(item in transaction for item in itemset)
                for transaction in df["Items"]
            )
            item_counts[itemset] += support_count
    else:
        for items in df["Items"]:
            for item in items:
                item_counts[item] += 1
    return item_counts


def generate_frequent_items(candidates, supp):
    return dict(
        sorted((key, value) for key, value in candidates.items() if value >= supp)
    )


dataset2_bis = getTransDataset(df, transactions)


def apriori(df, items, transactions, support):
    total_L = {}
    trans_dataset = getTransDataset(df, transactions)
    candidates = calculateSupport(items, trans_dataset)
    L = generate_frequent_items(candidates, support)
    total_L.update(L)

    k = 1
    while L:
        previous_dict = L
        associations = generate_k_associations(previous_dict, 2)
        candidates = calculateSupport(associations, trans_dataset, previous_dict, k)
        L = generate_frequent_items(candidates, support)
        total_L.update(L)
        k += 1
    return previous_dict, total_L


result, total_L = apriori(df, items, transactions, 2)


def generate_association_rules(associations):
    rules = set()
    for association, support in associations.items():
        items = list(association)
        for item in items:
            antecedent = (item,)
            consequent_candidates = [c for c in items if c != item]
            # Generate rules without repetitions
            for i in range(1, len(consequent_candidates) + 1):
                for combination in combinations(consequent_candidates, i):
                    rule = (antecedent, combination)
                    reversed_rule = (combination, antecedent)
                    rules.add(rule)
                    rules.add(reversed_rule)
    return list(rules)


rules = generate_association_rules(result)


def calculate_confidence(rules, frequent_items, threshold):
    rules_confidence = {}
    for rule in rules:
        antecedent, consequent = rule
        union = sorted(antecedent + consequent)
        antecedent = sorted(antecedent)

        # Convert antecedent to a single-item value if it's a single-item tuple
        antecedent_value = antecedent[0] if len(antecedent) == 1 else tuple(antecedent)

        confidence = frequent_items[tuple(union)] / frequent_items[antecedent_value]
        if confidence >= threshold:
            rules_confidence[rule] = confidence
    return rules_confidence


calculate_confidence(rules, total_L, 0.6)
