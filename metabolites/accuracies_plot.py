import matplotlib.pyplot as plt

# Model names and accuracies
models = ['Masrur et al.[9]', 'Before PCA ( Base paper )', 'After PCA ( Base paper )', 'Our Model']
serum_accuracies = [83.33,81.82,86.36,82.3]
plasma_accuracies = [87.5,90.91,78.03,88.235]


# Plotting
fig, ax = plt.subplots()
bar_width = 0.35
index = range(len(models))

serum_bars = ax.bar(index, serum_accuracies, bar_width, label='Serum')
plasma_bars = ax.bar([i + bar_width for i in index], plasma_accuracies, bar_width, label='Plasma')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Models for Serum and Plasma Samples')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(models)
ax.legend()

plt.show()
