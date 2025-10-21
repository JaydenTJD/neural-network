from keras.datasets import mnist
(train_data, train_answers), (test_data, test_answers) = mnist.load_data()
print("Training data shape:", train_data.shape, train_answers.shape)
print("Testing data shape:", test_data.shape, test_answers.shape)
print("Writing training data...")
with open("dataset/train_data.bin", "w") as file:
    file.write(','.join([','.join([','.join([str(p) for p in r]) for r in i]) for i in train_data]))
print("Done writing training data!")
print("Writing training answers...")
with open("dataset/train_answers.bin", "w") as file:
    file.write(','.join([str(a) for a in train_answers]))
print("Done writing training answers!")
print("Writing testing data...")
with open("dataset/test_data.bin", "w") as file:
    file.write(','.join([','.join([','.join([str(p) for p in r]) for r in i]) for i in test_data]))
print("Done writing testing data!")
print("Writing testing answers...")
with open("dataset/test_answers.bin", "w") as file:
    file.write(','.join([str(a) for a in test_answers]))
print("Done writing testing answers!")
print("All files extracted and written successfully!")