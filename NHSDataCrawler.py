import json
from bs4 import BeautifulSoup  # 网页解析，获取数据
import re  # 正则表达式，进行文字匹配`
import urllib.request, urllib.error  # 制定URL，获取网页数据

# store all the information
disease_data = []

# # 目标URL
# url = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/abdominal-aortic-aneurysm/"
pattern = re.compile(r'(<p>(.*?)</p>)|(<ul class="wp-block-list">(.*?)</ul>)', re.DOTALL)
end_pattern = re.compile(r'(<div class="no-print push--ends">)')

def askURL(url):
    # use urllib
    head = {  # 模拟浏览器头部信息，向豆瓣服务器发送消息
        "User-Agent": "Mozilla / 5.0(Windows NT 10.0; Win64; x64) AppleWebKit / 537.36(KHTML, like Gecko) Chrome / 80.0.3987.122  Safari / 537.36"
    }
    # 用户代理，表示告诉豆瓣服务器，我们是什么类型的机器、浏览器（本质上是告诉浏览器，我们可以接收什么水平的文件内容）
    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
    return html

# 读取文件并解析疾病名称和链接
disease_link = []
with open("nhs_disease_links.txt", "r", encoding="utf-8") as f:
    for line in f:
        # 只在第一个 `": "` 处进行拆分，避免URL中有 `:` 时出错
        parts = line.strip().rsplit(": ", 1)
        if len(parts) == 2:  # 确保分割后有两个部分
            disease, link = parts
            disease_link.append((disease, link))

for disease, url in disease_link:
    try:
        # 发送请求
        html = askURL(url)

        # 如果找到了end_pattern，则截取 HTML，仅保留其之前的部分
        end_match = end_pattern.search(html)
        if end_match:
            html = html[:end_match.start()]

        # 解析HTML
        soup = BeautifulSoup(html, "html.parser")

        # 当前匹配的标志，标记我们是否已遇到第一个 <p> 标签
        found_first_p = False

        description = []

        for match in pattern.finditer(html):
            if match.group(1):  # 如果是 <p> 标签
                if not found_first_p:
                    # 处理第一个 <p> 标签
                    found_first_p = True  # 标记我们已经找到了第一个 <p>
                p_text = re.sub(r"<.*?>", "", match.group(2).strip())  # 去除 HTML 标签
                description.append(p_text)  # 添加到 description 列表

            elif match.group(3) and found_first_p:  # 如果是 <ul> 标签，并且已找到第一个 <p>
                ul_content = match.group(4)
                li_items = re.findall(r"<li>(.*?)</li>", ul_content, re.DOTALL)
                for li in li_items:
                    li_text = re.sub(r"<.*?>", "", li.strip())  # 去除 HTML 标签
                    description.append(f"- {li_text}")  # 添加到 description，并格式化

        description_text = "\n".join(description)
        disease_data.append({"disease": disease, "description": description_text})

    except Exception as e:
        print(f"Error processing {disease}: {e}")

with open("disease_data.json", "w", encoding="utf-8") as f:
    json.dump(disease_data, f, ensure_ascii=False, indent=4)

print("数据成功保存到 disease_data.json！")

