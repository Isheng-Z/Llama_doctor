import requests
from bs4 import BeautifulSoup  # 网页解析，获取数据
import re  # 正则表达式，进行文字匹配`
import urllib.request, urllib.error  # 制定URL，获取网页数据

# 目标URL
url_origin = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/"
pattern_links = re.compile(r'<a href="https://www.nhsinform.scot/illnesses-and-conditions(.*?)">(.*?)</a>')

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

# 发送请求
html_base = askURL(url_origin)

# 解析HTML
soup = BeautifulSoup(html_base, "html.parser")

# 查找符合条件的链接
matches = pattern_links.findall(str(soup))

# 处理匹配结果
base_url = "https://www.nhsinform.scot/illnesses-and-conditions"
disease_data = [(name.strip(), base_url + link) for link, name in matches if name.strip() != "Illnesses and conditions"]

# 保存到文件
with open("nhs_disease_links.txt", "w", encoding="utf-8") as f:
    for disease, link in disease_data:
        f.write(f"{disease}: {link}\n")

print(f"共爬取 {len(disease_data)} 条疾病信息！")