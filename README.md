# Brain MRI Tumor Classifier

Educational deep learning project that classifies brain MRI slices into four classes:
glioma, meningioma, pituitary tumor, and no tumor.

> **Not a medical device. Not for diagnosis.** This is a learning project.

## Dataset

Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle). 5,600 training and 1,600 test images, balanced across four classes.
Validation was 20% of the Training folder; the Testing folder was kept untouched for final scores.

## Results (held-out test set, 1,600 images)

| Model                            | Test accuracy | Glioma recall | Tumors labeled "no tumor" |
| -------------------------------- | ------------- | ------------- | ------------------------- |
| Baseline CNN                     | 88.7%         | 0.78          | 76                        |
| CNN + augmentation + LR schedule | 89.6%         | 0.74          | 92                        |
| EfficientNetB0 (fine-tuned)      | **92.1%**     | 0.79          | **23**                    |

## Findings

- Validation accuracy (~96%) was higher than test accuracy (~92%). The validation split comes from the same folder as training and likely shares near-duplicate slices, so the test folder is the honest number.
- Transfer learning cut missed tumors from 92 to 23.
- Glioma is the weak class and is often confused with meningioma.
- Grad-CAM shows the model usually attends to brain tissue and lesions, but some errors have heatmaps on the lesion itself, so the issue is classification, not attention.
- Softmax scores are overconfident: wrong predictions can show 90%+.

## Run it

```bash
pip install -r requirements.txt
# place brain_tumor_effnet.keras in this folder
python app.py
```

## Limitations

2D slices only, single dataset, no patient-level split, no calibration, no external validation.
