import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Model': [
        'kykim/albert-kor-base', 'kykim/albert-kor-base', 'kykim/albert-kor-base', 'kykim/albert-kor-base',
        'klue/bert-base', 'klue/bert-base', 'klue/bert-base', 'klue/bert-base', 
        'klue/roberta-base', 'klue/roberta-base', 'klue/roberta-base', 'klue/roberta-base',
        'beomi/kcbert-base', 'beomi/kcbert-base', 'beomi/kcbert-base', 'beomi/kcbert-base', 
        'beomi/KcELECTRA-base-v2022', 'beomi/KcELECTRA-base-v2022', 'beomi/KcELECTRA-base-v2022', 'beomi/KcELECTRA-base-v2022'
    ],
    'Max Length': [
        16, 32, 64, 128,
        16, 32, 64, 128,
        16, 32, 64, 128,
        16, 32, 64, 128,
        16, 32, 64, 128
    ],
    'Accuracy': [
        0.8655346489215237, 0.9040844424047728, 0.9160165213400643, 0.9123451124368976,
        0.8251491509866912, 0.8893988067921065, 0.9082147774208352, 0.9196879302432308,
        0.8402937127122533, 0.8737953189536485, 0.9155575952271684, 0.9183111519045434,
        0.833409821018816, 0.8953648462597522, 0.914180816888481, 0.9063790729692519, 
        0.8558972005507114, 0.8981184029371271, 0.9210647085819184, 0.9242771913721891
    ],
    'Recall': [
        0.8655346489215237, 0.9040844424047728, 0.9160165213400643, 0.9123451124368976, 
        0.8251491509866912, 0.8893988067921065, 0.9082147774208352, 0.9196879302432308,
        0.8402937127122533, 0.8737953189536485, 0.9155575952271684, 0.9183111519045434, 
        0.833409821018816, 0.8953648462597522, 0.914180816888481, 0.9063790729692519, 
        0.8558972005507114, 0.8981184029371271, 0.9210647085819184, 0.9242771913721891
    ],
    'Precision': [
        0.8663138681956906, 0.9038478255391614, 0.9155749868374288, 0.9140891675454119, 
        0.8310323593355535, 0.8903438856377658, 0.909098453019656, 0.9201945770118346, 
        0.8390627239605588, 0.8732560170362128, 0.9153800337095365, 0.9184015157832663, 
        0.8393343385028657, 0.8958494429757468, 0.9144406841856689, 0.9062191794479928, 
        0.858374344971045, 0.9020071391926895, 0.9212278799286411, 0.9239260782717686
    ],
    'F1-Score': [
        0.8619471962942978, 0.9039494248094994, 0.9155349923860906, 0.9105686475476488,
        0.8269796357217374, 0.8870101828017759, 0.9085391343317589, 0.9185663677176941, 
        0.8363712163326771, 0.8715644256052623, 0.9146906506041333, 0.9173474837424787,
        0.8351909118810812, 0.8934484018201772, 0.9130166588741929, 0.9062916886770098, 
        0.850788095605706, 0.8951287482567445, 0.9201335236163277, 0.923921243299869
    ],
    'Epochs Trained': [
        12, 13, 11, 12,
        11, 11, 12, 12,
        12, 12, 12, 13, 
        12, 12, 12, 11, 
        12, 12, 12, 12
    ],
    'Average Memory Usage (MB)': [
        165893.02083333334, 173179.78846153847, 181046.81818181818, 183342.3125, 
        1343479.9318181819, 1355026.0909090908, 1409345.0, 1434165.0208333333, 
        1426852.5625, 1363567.1458333333, 1388385.5208333333, 1397415.923076923, 
        1399659.3958333333, 1331035.125, 1392244.1666666667, 1397315.8181818181,
        1590798.7916666667, 1580752.5833333333, 1625487.1041666667, 1662332.6875
    ]
}

memory_data = {
    'Max Length': [16, 32, 64, 128],
    'kykim/albert-kor-base': [165893.02083333334, 173179.78846153847, 181046.81818181818, 183342.3125],
    'klue/bert-base': [1343479.9318181819, 1355026.0909090908, 1409345.0, 1434165.0208333333],
    'klue/roberta-base': [1426852.5625, 1363567.1458333333, 1388385.5208333333, 1397415.923076923],
    'beomi/kcbert-base': [1399659.3958333333, 1331035.125, 1392244.1666666667, 1397315.8181818181],
    'beomi/KcELECTRA-base-v2022': [1590798.7916666667, 1580752.5833333333, 1625487.1041666667, 1662332.6875]
}


df = pd.DataFrame(data)
memory_df = pd.DataFrame(memory_data)


plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (18, 6),
    'axes.grid': True,
    'grid.alpha': 0.6,
    'grid.linestyle': '--',
    'legend.loc': 'best',
    'legend.title_fontsize': 12
})

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
plt.subplots_adjust(wspace = 5)


filtered_df = df[df['Max Length'].isin([16, 32, 64, 128])]


for model in filtered_df['Model'].unique():
    subset = filtered_df[filtered_df['Model'] == model]
    axes[0].plot(subset['Max Length'], subset['Accuracy'], marker='o', label=model.split('/')[1])

axes[0].set_title('Accuracy by Model and Max Length')
axes[0].set_xlabel('Max Length')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticks([16, 32, 64, 128])
axes[0].legend(title='Model')
axes[0].grid(True)

for model in memory_df.columns[1:]:
    axes[1].plot(memory_df['Max Length'], memory_df[model], marker='o', label=model.split('/')[1])

axes[1].set_title('Memory Usage by Model and Max Length')
axes[1].set_xlabel('Max Length')
axes[1].set_ylabel('Memory Usage (MB)')
axes[1].set_xticks([16, 32, 64, 128])
axes[1].legend(title='Model')
axes[1].grid(True)

plt.tight_layout()
plt.show()
