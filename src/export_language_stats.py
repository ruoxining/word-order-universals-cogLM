import os
import math
import json
import pandas as pd
from collections import defaultdict

def main():
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    df = pd.read_csv("data/language.csv")
    df = df.rename(columns={
                '51A Position of Case Affixes': 'Case', 
                '82A Order of Subject and Verb': 'SV', 
                '83A Order of Object and Verb': 'OV',
                '85A Order of Adposition and Noun Phrase': 'PP',
                '87A Order of Adjective and Noun': 'Adj',
                '90A Order of Relative Clause and Noun': 'Rel',
                    })

    case2switch = {'1 Case suffixes': 0, '2 Case prefixes': 1}
    SV2switch = {'1 SV': 0, '2 VS': 1}
    OV2switch = {'1 OV': 0, '2 VO': 1}
    PP2switch = {'2 Prepositions': 1, '1 Postpositions': 0}
    Adj2switch = {'2 Noun-Adjective': 1, '1 Adjective-Noun': 0}
    Rel2switch = {'1 Noun-Relative clause': 1, '2 Relative clause-Noun': 0}

    df['Case'] = df['Case'].map(case2switch)
    df['SV'] = df['SV'].map(SV2switch)
    df['OV'] = df['OV'].map(OV2switch)
    df['PP'] = df['PP'].map(PP2switch)
    df['Adj'] = df['Adj'].map(Adj2switch)
    df['Rel'] = df['Rel'].map(Rel2switch)

    target_columns = [
    'Case',
    'SV',
    'OV',
    'PP',
    'Adj',
    'Rel',
    ]

    print("all languges:", len(df))

    lang_filter = df[target_columns].isna().sum(axis=1)<6
    df = df[lang_filter]
    switch_df = df[target_columns]
    print("languages with at least one switch:", len(df))
    print("all switches:", switch_df.size)
    print("filled switches:", switch_df.size - (switch_df.isna()).sum().sum())

    switch2position = {"Case": 6, "SV": 0, "OV": 1, "PP": 3, "Adj": 4, "Rel": 5}
    lang2score = defaultdict(float)

    for row in switch_df.iterrows():
        langs = []
        switches = dict(row[1])
        n_nan = sum([math.isnan(s) or s==0.5 for s in switches.values()])
        if n_nan:
            nan_possible_switch = [list(format(i, f'0{str(n_nan)}b')) for i in range(2**n_nan)]
            for ns in nan_possible_switch:
                lang_id = ["2"]*7
                for k, s in switches.items():
                    if math.isnan(s) or s == 0.5:
                        s = ns.pop(0)
                    lang_id[switch2position[k]] = str(int(s))
                lang2score["".join(lang_id)] += 1/(2**n_nan)
        else:
            lang_id = "".join([str(int(switches["SV"])), str(int(switches["OV"])), "2", str(int(switches["PP"])), str(int(switches["Adj"])), str(int(switches["Rel"])), str(int(switches["Case"]))])
            lang2score[lang_id] += 1
            
    print("SOV", sum([v for k, v in lang2score.items() if k[:2] == "00"])/sum(lang2score.values()))
    print("SVO", sum([v for k, v in lang2score.items() if k[:2] == "01"])/sum(lang2score.values()))
    print("OVS", sum([v for k, v in lang2score.items() if k[:2] == "10"])/sum(lang2score.values()))
    print("VOS", sum([v for k, v in lang2score.items() if k[:2] == "11"])/sum(lang2score.values()))

    marginal = sum(float(v) for v in lang2score.values())
    marginalized_scores = {k: float(v)/marginal for k, v in lang2score.items()}
    json.dump(marginalized_scores, open("work/lang2count.json", "w"))
    print("file dumped!")


if __name__ == "__main__":
    main()