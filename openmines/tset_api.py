# test_api.py

import openai
import json

def test_openai_api():
    # 配置 API
    openai.api_key = "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"  # 替换成你的实际 API 密钥
    openai.api_base = "https://api.openai.com/v1"

    print("开始测试 OpenAI API...")
    
    try:
        # 发送测试请求
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Say this is a test!"}
            ],
            temperature=0.7
        )
        
        # 打印响应
        print("\nAPI 响应成功！")
        print("\n完整响应:")
        print(json.dumps(response, indent=4))
        print("\n响应内容:", response.choices[0].message.content)
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    test_openai_api()