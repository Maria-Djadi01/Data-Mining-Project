import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
from itertools import combinations
import sys
import matplotlib.pyplot as plt
import os


project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_directory)


# sys.path.append("../../../Data-Mining-Project")
from src.utils import plot_apriori_results

# ----------------------------------------------------------------#
# Load data #
# ----------------------------------------------------------------#

# df = pd.read_csv("../data/processed/static_dataset3_discretized.csv", index_col=0)

# apriori_df = df.drop(
#     columns=[
#         "Temperature",
#         "Humidity",
#         "Rainfall",
#         "Temperature_width_disc",
#         "Humidity_width_disc",
#         "Rainfall_width_disc",
#     ]
# )

# ----------------------------------------------------------------#
# Apriori Algorithm
# ----------------------------------------------------------------#


def getTransDataset(df, transactions):
    df_result = pd.DataFrame(columns=["Transaction", "Items"])

    for trans in transactions:
        itemsList = df.iloc[trans]
        df_result.loc[len(df_result)] = {
            "Transaction": trans,
            "Items": list(set(itemsList)),
        }

    return df_result


# Generate C
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


def generate_association_rules(associations):
    rules = set()
    for association, support in associations.items():
        items = []
        if isinstance(association, tuple):
            for asso in association:
                items.append(asso)
        else:
            items.append(association)
        # items = list(association)
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


def apriori(df, apriori_df, minSupp, minConf):
    transactions = apriori_df.index

    items = []
    for col in apriori_df.columns:
        items.append(apriori_df[col].unique())

    # Calculate min supp
    # supp_min = int(minSupp * len(transactions) / 100)
    # conf_min = minConf / 100

    total_L = {}
    trans_dataset = getTransDataset(df, transactions)
    candidates = calculateSupport(items, trans_dataset)
    L = generate_frequent_items(candidates, minSupp)
    total_L.update(L)

    k = 1
    while L:
        previous_dict = L
        associations = generate_k_associations(previous_dict, 2)
        candidates = calculateSupport(associations, trans_dataset, previous_dict, k)
        L = generate_frequent_items(candidates, minSupp)
        total_L.update(L)
        k += 1
    rules = generate_association_rules(total_L)
    result = calculate_confidence(rules, total_L, minConf)
    return total_L, rules, result


# total_L, rules, result = apriori(df, apriori_df, 10, 60)
# print(total_L)


def perform_experiments(df, min_supp_range, min_conf_range):
    experiment_results = []

    for min_supp in min_supp_range:
        for min_conf in min_conf_range:
            total_L, rules, result = apriori(df, apriori_df, min_supp, min_conf)

            experiment_results.append(
                {
                    "Min_Supp": min_supp,
                    "Min_Conf": min_conf,
                    "Total_L_Count": len(total_L),
                    "Rules_Count": len(rules),
                    "Result_Count": len(result),
                }
            )

    return experiment_results


min_supp_range = range(5, 21)
min_conf_range = range(40, 100, 10)

# experiment_results = perform_experiments(apriori_df, min_supp_range, min_conf_range)


def plot_min_supp_vs_frequent_items(experiment_results):
    sns.set(style="whitegrid")

    min_supp_values = [result["Min_Supp"] for result in experiment_results]
    frequent_items_count_values = [
        result["Total_L_Count"] for result in experiment_results
    ]

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        x=min_supp_values,
        y=frequent_items_count_values,
        marker="o",
        label="Frequent Items Count",
    )
    plt.xlabel("Min_Supp")
    plt.ylabel("Frequent_Items_Count")
    plt.title("Min_Supp vs Frequent_Items_Count")
    plt.legend()
    plt.show()


# plot_min_supp_vs_frequent_items(experiment_results)


def plot_min_supp_vs_rules_count(experiment_results):
    sns.set(style="whitegrid")

    min_supp_values = [result["Min_Supp"] for result in experiment_results]
    result_count_values = [result["Result_Count"] for result in experiment_results]

    print("Length of min_supp_values:", len(min_supp_values))
    print("Length of result_count_values:", len(result_count_values))

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        x=min_supp_values, y=result_count_values, marker="o", label="Rules Count"
    )
    plt.xlabel("Min_Supp")
    plt.ylabel("Rules_Count")
    plt.title("Min_Supp vs Rules_Count")
    plt.legend()
    plt.show()


# plot_min_supp_vs_rules_count(experiment_results)


def plot_min_conf_vs_rules_count(experiment_results):
    sns.set(style="whitegrid")

    min_supp_range = range(5, 21)

    plt.figure(figsize=(15, 5))

    for min_supp_value in min_supp_range:
        result_subset = [
            result
            for result in experiment_results
            if result["Min_Supp"] == min_supp_value
        ]
        result_subset = sorted(result_subset, key=lambda x: x["Min_Conf"])

        conf_values = [result["Min_Conf"] for result in result_subset]
        result_count_values = [result["Result_Count"] for result in result_subset]

        plt.plot(
            conf_values,
            result_count_values,
            label=f"Min_Supp={min_supp_value}",
            marker="o",
        )

    plt.xlabel("Min_Conf")
    plt.ylabel("Rules_Count")
    plt.title("Rules_Count vs Min_Conf (grouped by Min_Supp)")
    plt.legend(title="Min_Supp", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


# plot_min_conf_vs_rules_count(experiment_results)


# ----------------------------------------------------------------#
# Strong associations Extract (Confidence = 1.0)
# ----------------------------------------------------------------#


# strong_rules = [rule for rule, confidence in result.items() if confidence == 1.0]
