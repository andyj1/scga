import torch
from typing import Dict

def compute_adv_scores(clean_preds, adv_preds, labels) -> Dict[str, float]:
    clean_preds = torch.tensor(clean_preds)
    adv_preds = torch.tensor(adv_preds)
    labels = torch.tensor(labels)

    originally_correct = clean_preds == labels
    originally_wrong = clean_preds != labels

    asr_numerator = torch.sum((adv_preds != labels) & originally_correct).item()
    asr_denominator = torch.sum(originally_correct).item()
    asr = (asr_numerator / asr_denominator * 100) if asr_denominator > 0 else 0.0

    acr_numerator = torch.sum((adv_preds == labels) & originally_wrong).item()
    acr_denominator = torch.sum(originally_wrong).item()
    acr = (acr_numerator / acr_denominator * 100) if acr_denominator > 0 else 0.0

    fooling_rate = (torch.sum(clean_preds != adv_preds) / clean_preds.shape[0] * 100).item()

    acc_clean = (torch.sum(clean_preds == labels) / clean_preds.shape[0] * 100).item()
    acc_adv = (torch.sum(adv_preds == labels) / adv_preds.shape[0] * 100).item()

    return {
        "CleanAcc": round(acc_clean, 4),
        "AdvAcc": round(acc_adv, 4),
        "ASR": round(asr, 4),
        "ACR": round(acr, 4),
        "FR": round(fooling_rate, 4),
        "NumCleanCorrect": int(asr_denominator),
        "NumCleanIncorrect": int(acr_denominator),
        "NumAdvIncorrect": int(asr_numerator),
        "NumAdvCorrect": int(acr_numerator)
    }


if __name__ == "__main__":
    # Sample usage structure
    import numpy as np
    sample_clean_preds = np.random.randint(0, 10, 10)  # Dummy predicted labels (clean)
    sample_adv_preds = np.random.randint(0, 10, 10)    # Dummy predicted labels (adv)
    sample_labels = np.random.randint(0, 10, 10)       # Ground truth labels
    print(sample_clean_preds)
    print(sample_adv_preds)
    print('--'*20)
    print(sample_labels)

    # Compute

    results = compute_asr_acr_fooling_rate_torch(sample_clean_preds, sample_adv_preds, sample_labels)
    from prettytable import PrettyTable

    # Create a PrettyTable instance
    table = PrettyTable()

    # Define the column names
    table.field_names = ['Model Name'] + list(results.keys())
    
    table.add_row(['Model Name'] + [f'{value:.4f}' for value in list(results.values())])
    # Add rows to the table
    # for metric, value in results.items():
    #     print(metric, value)
    #     table.add_row([metric, f'{value:.4f}'])
    # table.add_row(["Top1 Clean", f'{results["CleanAcc"]:.4f}'])
    # table.add_row(["Top1 Adv", f'{results["AdvAcc"]:.4f}'])
    # table.add_row(["ASR", f'{results["ASR"]:.4f}'])
    # table.add_row(["ACR", f'{results["ACR"]:.4f}'])
    # table.add_row(["Fooling Rate", f'{results["FR"]:.4f}'])
    # table.add_row(["Clean Correct", results["NumCleanCorrect"]])
    # table.add_row(["Clean Incorrect", results["NumCleanInCorrect"]])
    # total_correct_incorrect = results["NumCleanCorrect"] + results["NumCleanInCorrect"]
    # table.add_row(["Total Correct + Incorrect", total_correct_incorrect])


    print(table)
