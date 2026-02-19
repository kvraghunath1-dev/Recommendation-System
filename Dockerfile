# olg-reco-container/Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# 1) Python deps
COPY olg-reco-container/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
# before the pip install line, add:
RUN yum install -y gcc gcc-c++ libgomp && yum clean all && rm -rf /var/cache/yum
RUN python -m pip install --no-cache-dir implicit==0.7.2

RUN python -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# 2) App code
COPY olg-reco-container/handler.py ${LAMBDA_TASK_ROOT}/handler.py
COPY olg-reco-container/recommender.py ${LAMBDA_TASK_ROOT}/recommender.py

# 3) Model artifacts
COPY artifacts ${LAMBDA_TASK_ROOT}/artifacts

# 4) Config + entry
ENV MODEL_PATH=/var/task/artifacts/model.joblib
CMD ["handler.lambda_handler"]
