 ## run.py 파일 
 
 ## 데이터 전처리 부분
from imageatm.components import DataPrep
dp = DataPrep(
    image_dir='./train',
    samples_file='data.json',
    job_dir= './sample'
)
dp.run(resize=True)

## 모델을 학습시키는 부분
from imageatm.components import Training
trainer = Training(dp.image_dir, dp.job_dir,
epochs_train_dense=3, epochs_train_all=1)
trainer.run()

## 학습모델의 평가 부분
from imageatm.components import Evaluation
e = Evaluation(image_dir=dp.image_dir, job_dir=dp.job_dir)
e.run()
