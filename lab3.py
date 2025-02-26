from fastapi import FastAPI
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # 导入BaseModel用于定义请求体模型


app = FastAPI()

# 碰到的问题，由于未开启跨域请求，接收到内容为
# INFO:     127.0.0.1:53482 - "OPTIONS /analyze/ HTTP/1.1" 405 Method Not Allowed
# INFO:     127.0.0.1:53493 - "GET / HTTP/1.1" 404 Not Found
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_methods=["*"],  # 允许所有方法(GET,POST,OPTIONS等)
    allow_headers=["*"],  # 允许所有头
)


classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


# 碰到的问题，在html中输入内容，请求的数据格式不正确，导致服务器无法处理，接收到的内容为
# INFO:     127.0.0.1:53883 - "OPTIONS /analyze/ HTTP/1.1" 200 OK
# INFO:     127.0.0.1:53883 - "POST /analyze/ HTTP/1.1" 422 Unprocessable Entity
# step1 确保.html文件中的fetch请求发送JSON数据正确是body: JSON.stringify({text: text})
# step2 在Py中加入BaseModel用于定义请求体模型，并且使用请求体模型，从请求体中提取text
class TextRequest(BaseModel):
    text: str


@app.post("/analyze/")
def analyze_text(request: TextRequest):
    result = classifier(request.text)
    return {"label": result[0]['label'], "score": result[0]['score']}


if __name__ == "__main__":
    import uvicorn
    print(uvicorn.run(app, host="0.0.0.0", port=3000))