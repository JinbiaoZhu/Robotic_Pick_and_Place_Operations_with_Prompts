"""
Description: Basic tools for code running.
Editor: Jinbiao Zhu
Date: 15-01-2024
"""

import os
import json


def read_and_average(folder_path, key):
    values = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                try:
                    value = data[key]
                    values.append(value)
                except KeyError:
                    print(f"Key '{key}' not found in {filename}.")
                except ValueError:
                    print(f"Error reading {filename}.")

    if values:
        average = sum(values) / len(values)
        return average
    else:
        return None


def read_and_sum(folder_path, key):
    values = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                try:
                    value = data[key]
                    values.append(value)
                except KeyError:
                    print(f"Key '{key}' not found in {filename}.")
                except ValueError:
                    print(f"Error reading {filename}.")

    if values:
        average = sum(values)
        return average
    else:
        return None

