import re
from typing import List, Callable
from transformers import pipeline

# 深度学习的模型初始化
# 初始化NER的pipeline
ner_pipeline = pipeline(
    "ner",
    model="ckiplab/bert-base-chinese-ner",
    aggregation_strategy="simple"
)

def ner_mask(text: str) -> str:
    # 使用深度学习模型进行语义级别的脱敏（包括人名和地名）
    entities = ner_pipeline(text)
    spans = []

    # 提取模型识别的实体和位置
    for ent in entities:
        label = ent["entity_group"]
        start = ent["start"]
        end = ent["end"]

        # 将识别的实体类型映射到占位符
        if label == "PER":
            spans.append((start, end, "[NAME]"))
        elif label == "LOC":
            spans.append((start, end, "[PLACE]"))
    
    # 先按照起始的位置排序，如果相同再按照长度排序(先处理长的实体)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # 去除重叠或包含关系的实体区间
    filtered_spans = []
    last_end = -1
    for start, end, tag in spans:
        if start >= last_end: # 当前的实体起始位置在上一实体结束之后才保留
            filtered_spans.append((start, end, tag))
            last_end = end
    
    # 重建文本
    result = []
    last_idx = 0
    for start, end, tag in filtered_spans:
        result.append(text[last_idx:start]) # 添加实体前的文本
        result.append(tag) # 添加占位符
        last_idx = end # 更新最后处理的位置
    result.append(text[last_idx:]) # 补充剩下的文本

    return ''.join(result)


# 脱敏流水线架构
class DesensitizationPipeline:
    # 允许按照顺序添加多个处理的步骤
    def __init__(self):
        self.steps: List[Callable[[str], str]] = []

    def add_step(self, func: Callable[[str], str]):
        # 处理环节（正则替换、NER替换等）
        self.steps.append(func)

    def run(self, text: str) -> str:
        # 依次执行所有步骤 
        for step in self.steps:
            text = step(text)
        return text
    

# 具体的步骤
def normalize_text(text: str) -> str:
    # 去掉首尾空格
    return text.strip()

# 高确定性的规则
def mask_phone(text: str) -> str:
    # 正则匹配手机号+86
    return re.sub(r'1[3-9]\d{9}', '[PHONE]', text)

def mask_email(text: str) -> str:
    # 正则匹配邮箱
    return re.sub(r'[a-zA-Z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)

# 中确定性的规则（关键词上下文）
def mask_address(text: str) -> str:
    # 通过居住于等一些关键词地址匹配
    return re.sub(
        r'(居住于|现居住于|地址|现居于)([\u4e00-\u9fa5A-Za-z0-9]+)',
        r'\1[PLACE]',
        text
    )

# 低确定性规则（语法结构的简单托底）
def mask_name(text: str) -> str:
    # 匹配句首或者标点之后的“xxx的”结构， 放在NER后面作为补充
    return re.sub(
        r'(?:(?<=^)|(?<=[，。！？；]))([\u4e00-\u9fa5]{2,3})(的)',
        r'[NAME]\2',
        text
    )

def clean_punctuation(text: str) -> str:
    # 去掉多余的标点符号
    return re.sub(r'[^\w\s\[\]]+', '', text)

# 构建并测试
def build_pipeline():
    # 组装流水线：按照“预处理->高确定性正则->AI识别->低准确率正则兜底”的顺序
    p = DesensitizationPipeline()

    # 基础清洗
    p.add_step(normalize_text)

    # 先处理手机、邮箱类的高确定性信息
    p.add_step(mask_phone)
    p.add_step(mask_email)

    # AI模型识别的语义级别脱敏
    p.add_step(ner_mask)

    # 最后处理一些低确定性的规则
    p.add_step(mask_address)
    p.add_step(mask_name)

    # 清理多余的标点
    p.add_step(clean_punctuation)

    return p

if __name__ == "__main__":
    pipeline = build_pipeline()

    # 测试文本
    test_text = "张三的手机号是13812345678，邮箱是H6Bm7@example.com，居住于北京市海淀区。"
    result = pipeline.run(test_text)
    print("----------系统测试----------")
    print("原始文本：", test_text)
    print("脱敏结果：", result)