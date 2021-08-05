import pandas as pd


def convert(image_f, label_f, out_f, n):
    f = open(image_f, "rb")
    o = open(out_f, "w")
    l = open(label_f, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


convert("data/train_images.idx3-ubyte", "data/train_labels.idx1-ubyte",
        "data/train.csv", 60000)
convert("data/test_images.idx3-ubyte", "data/test_labels.idx1-ubyte",
        "data/test.csv", 10000)

print("converted mnist to csv")

labels = ["label"]
for i in range(784):
    labels.append("pixel" + str(i))
train_dataset = pd.read_csv("data/train.csv")
test_dataset = pd.read_csv("data/test.csv")
train_dataset.to_csv("data/train1.csv", header=labels, index=False)
train_dataset.to_csv("data/test1.csv", header=labels, index=False)
