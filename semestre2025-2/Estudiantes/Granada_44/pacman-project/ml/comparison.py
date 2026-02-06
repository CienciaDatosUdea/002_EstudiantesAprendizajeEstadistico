import matplotlib.pyplot as plt
import pandas as pd
import os 

supervised_scores = [ 1373.0, -457.0, 85.0, 90.0, -136.0, -457.0, 100.0, 336.0, -293.0, -12.0, 1179.0, -1.0, 23.0, -12.0, -191.0, 344.0, 311.0, -380.0, -55.0, -91.0, 94.0, -457.0, 1175.0, 1182.0, 982.0, -131.0, -457.0, -64.0, -52.0, -133.0, 64.0, -113.0, 983.0, 72.0, 159.0, 981.0, 119.0, -140.0, -323.0, 1383.0, 1377.0, 108.0, -457.0, 326.0, 1382.0, 1149.0, 112.0, 72.0, 8.0, 430.0, 435.0, 337.0, -150.0, 1182.0, -457.0, 56.0, 21.0, 29.0, -176.0, -457.0, 44.0, 238.0, 75.0, 82.0, 113.0, 982.0, 101.0, 145.0, 16.0, -457.0, 983.0, 83.0, 1342.0, 1383.0, 1152.0, 42.0, 73.0, 1181.0, 522.0, 441.0, 655.0, 545.0, -167.0, -384.0, 76.0, -77.0, -73.0, -64.0, -457.0, -64.0, -157.0, 94.0, -124.0, 675.0, 334.0, -208.0, 4.0, -386.0, 229.0, -342.0]
supervised_wins_raw = ["Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Win", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Win", "Win", "Loss", "Loss", "Loss", "Win", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Win", "Win", "Win", "Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss"]

qlearning_scores = [-358.0, 982.0, 979.0, -299.0, -119.0, 957.0, -299.0, 932.0, -429.0, -265.0, 939.0, -365.0, -302.0, -244.0, -383.0, -112.0, -356.0, -60.0, -291.0, -240.0, -331.0, 978.0, 970.0, -211.0, -160.0, -149.0, 986.0, -270.0, -447.0, 978.0, 977.0, -271.0, -228.0, -303.0, -340.0, -368.0, -302.0, 978.0, -68.0, -81.0, -428.0, -365.0, -356.0, 982.0, -341.0, -287.0, 978.0, 971.0, -323.0, -446.0, -438.0, -148.0, 148.0, -285.0, -319.0, -159.0, 980.0, -91.0, 964.0, -303.0, -385, -303.0, -385.0, -446.0, -314.0, -221.0, -325.0, 973.0, -175.0, 985.0, -446.0, -295.0, -92.0, -383.0, -119.0, -302.0, -392.0, 980.0, -410.0, -87.0, -446.0, -101.0, -125.0, -422.0, 978.0, -311.0, -96.0, 930.0, -368.0, -302.0, -173.0, -111.0, 949.0, -158.0, -235.0, -282.0, -311.0, 901.0, -33.0, -271.0]
qlearning_wins_raw = ["Loss", "Win", "Win", "Loss", "Loss", "Win", "Loss", "Win", "Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Win", "Loss", "Loss", "Loss", "Win", "Loss", "Loss", "Win", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Loss", "Win", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Loss", "Loss", "Loss", "Win", "Loss", "Loss"]

random_scores = [-448.0, -466.0, -467.0, -481.0, -473.0, -446.0, -445.0, -356.0, -461.0, -425.0, -453.0, -385.0, -484.0, -464.0, -462.0, -384.0, -482.0, -131.0, -446.0, -491.0, -437.0, -424.0, -478.0, -441.0, -345.0, -392.0, -450.0, -446.0, -466.0, -413.0, -383.0, -483.0, -452.0, -506.0, -487.0, -459.0, -376.0, -474.0, -442.0, -354.0, -475.0, -400.0, -369.0, -375.0, -441.0, -482.0, -459.0, -377.0, -482.0, -462.0, -436.0, -443.0, -420.0, -135.0, -432.0, -15.0, -387.0, -418.0, -209.0, 53.0, -443.0, -426.0, -450.0, -430.0, -433.0, -449.0, -462.0, -422.0, -484.0, -18.0, -411.0, -449.0, -479.0, -460.0, -494.0, -458.0, -457.0, -480.0, -378.0, -469.0, -462.0, -473.0, -338.0, -423.0, -429.0, -483.0, -455.0, -462.0, -468.0, -440.0, -456.0, -477.0, -467.0, -431.0, -480.0, -482.0, -383.0, -417.0, -425.0, -482.0]
random_wins_raw = ["Loss"] * len(random_scores)


def create_safe_df(scores, wins_raw):
    min_len = min(len(scores), len(wins_raw))
    scores_trunc = scores[:min_len]
    wins_trunc = wins_raw[:min_len]
    wins_bin = [1 if x == "Win" else 0 for x in wins_trunc]
    return pd.DataFrame({'Score': scores_trunc, 'Win': wins_bin})

df_sup = create_safe_df(supervised_scores, supervised_wins_raw)
df_q = create_safe_df(qlearning_scores, qlearning_wins_raw)
df_rand = create_safe_df(random_scores, random_wins_raw)

#Moving average (so it looks okay)
window = 10
df_sup['MA_Score'] = df_sup['Score'].rolling(window=window).mean()
df_q['MA_Score'] = df_q['Score'].rolling(window=window).mean()
df_rand['MA_Score'] = df_rand['Score'].rolling(window=window).mean()

plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

#Scores
axes[0].plot(df_sup['MA_Score'], label='Supervised', color='blue', linewidth=2)
axes[0].plot(df_q['MA_Score'], label='Q-Learning', color='orange', linewidth=2, linestyle='--')
axes[0].plot(df_rand['MA_Score'], label='Random', color='gray', linewidth=2, linestyle=':')
axes[0].set_title(f'Tendencia de Puntaje')
axes[0].set_xlabel('Episodios')
axes[0].set_ylabel('Score Promedio')
axes[0].legend()
axes[0].grid(True)

#Win rates
win_rate_sup = (df_sup['Win'].sum() / len(df_sup)) * 100
win_rate_q = (df_q['Win'].sum() / len(df_q)) * 100
win_rate_rand = (df_rand['Win'].sum() / len(df_rand)) * 100

labels = ['Random', 'Supervised', 'Q-Learning']
rates = [win_rate_rand, win_rate_sup, win_rate_q]
colors = ['gray', 'blue', 'orange']

bars = axes[1].bar(labels, rates, color=colors, alpha=0.8)
axes[1].set_title('Tasa de Victorias (Win Rate %)')
axes[1].set_ylabel('Victorias (%)')
axes[1].set_ylim(0, max(rates) + 10)

for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()

nombre_archivo = 'comparativa_modelos_pacman.png'
ruta_completa = os.path.abspath(nombre_archivo)

plt.savefig(nombre_archivo, dpi=300)

plt.show()

#Print some data
print("\n--- Métricas Finales ---")
print(f"Supervised - Max: {max(df_sup['Score'])}, Avg: {df_sup['Score'].mean():.2f}, Win Rate: {win_rate_sup:.1f}%")
print(f"Q-Learning - Max: {max(df_q['Score'])}, Avg: {df_q['Score'].mean():.2f}, Win Rate: {win_rate_q:.1f}%")
print(f"Random     - Max: {max(df_rand['Score'])}, Avg: {df_rand['Score'].mean():.2f}, Win Rate: {win_rate_rand:.1f}%")