
from google import genai
from google.genai import types
import chromadb
import os
import sys
from dotenv import load_dotenv
from bidi.algorithm import get_display
import arabic_reshaper

load_dotenv(override=True)

# وظيفة مساعدة لطباعة النص العربي بشكل صحيح في الطرفية
def print_arabic(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    print(bidi_text)

# --- إعداد مفتاح API ---
# للحصول على مفتاح، قم بزيارة: https://aistudio.google.com/app/apikey
# ملاحظة: من الأفضل دائمًا استخدام متغيرات البيئة أو خدمات إدارة الأسرار لتخزين مفاتيح API بدلاً من كتابتها مباشرة في الكود.
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print_arabic("خطأ: لم يتم العثور على متغير البيئة GEMINI_API_KEY.")
    print_arabic("يرجى تعيينه لتشغيل هذا البرنامج النصي.")
    print_arabic("مثال على التعيين في نظام التشغيل الخاص بك:")
    print_arabic("export GEMINI_API_KEY='YOUR_API_KEY_HERE'")
    sys.exit(1) # الخروج من البرنامج إذا لم يتم العثور على المفتاح

client = genai.Client(api_key=api_key)

# لإعداد ChromaDB في الذاكرة لأغراض العرض التوضيحي.
# في بيئة حقيقية، ستقوم بتهيئة العميل للاتصال بقاعدة بيانات متجهة دائمة.
db = chromadb.Client()

# إذا كانت المجموعة موجودة مسبقاً، قم بإزالتها لضمان بداية نظيفة.
try:
    db.delete_collection(name="my_documents_collection")
except:
    pass

# إنشاء مجموعة جديدة للمستندات
collection = db.create_collection(name="my_documents_collection")

# --- إعداد المستندات المراد فهرسها ---
# هذه هي المستندات التي سيبحث فيها النموذج للإجابة على الأسئلة
documents = [
    "شركة النور حققت أرباحاً صافية بلغت 100 مليون دولار في الربع الأول من عام 2024. منتجها الأحدث هو 'النور برو', وهو حل برمجي لإدارة الطاقة.",
    "تأسست شركة النور في عام 2000، وتتخصص في تطوير حلول الطاقة المتجددة المبتكرة. مقرها الرئيسي في دبي.",
    "أرباح الربع الثاني لشركة النور في 2024 بلغت 120 مليون دولار، بزيادة قدرها 20% عن الربع السابق. يتميز 'النور برو' بواجهة مستخدم سهلة ودعم للذكاء الاصطناعي.",
    "الرئيس التنفيذي لشركة النور هو السيد أحمد السلمان، وقد أعلن عن خطط توسع عالمية في مؤتمر صحفي."
]

# --- الخطوة الأولى: تحويل المستندات إلى Embeddings (الفهرسة) ---
result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=documents)

print(result.embeddings)
embeddings = [e.values for e in result.embeddings]

# تخزين الـ Embeddings والمستندات الأصلية في قاعدة البيانات المتجهة (ChromaDB)
collection.add(
    embeddings=embeddings, # المتجهات الرقمية للمستندات
    documents=documents,   # المستندات الأصلية
    ids=[f"doc_{i}" for i in range(len(documents))] # معرّفات فريدة لكل مستند
)

print_arabic(f"تم تحويل {len(documents)} مستندات إلى Embeddings وحفظها في قاعدة البيانات!")
print_arabic("-" * 50)

# --- الخطوة الثانية: مرحلة السؤال والجواب (Retrieval Augmented Generation - RAG) ---

# السؤال الذي يطرحه المستخدم
# user_query = "كم بلغت أرباح شركة النور؟ وما هو أحدث منتجاتها؟"
user_query = "متى تأسست شركة النور ومن هو رئيسها التنفيذي؟"
print_arabic(f"سؤال المستخدم: {user_query}")

# 1. ترجمة السؤال إلى Embedding (استعلام الفهرس)
print_arabic("جاري تحويل سؤال المستخدم إلى Embedding...")
# توليد الـ Embedding لسؤال المستخدم
# يتم تعيين task_type كـ "RETRIEVAL_QUERY" لأنه استعلام للبحث
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=user_query,
)
query_embedding = result.embeddings[0].values
print_arabic("تم تحويل السؤال إلى Embedding.")

# 2. البحث (Retrieval) عن المستندات ذات الصلة
print_arabic("جاري البحث عن المستندات ذات الصلة في قاعدة البيانات...")
# استخدام الـ Embedding الخاص بالسؤال للبحث في قاعدة البيانات المتجهة
# عن المستندات الأكثر تشابهاً (الأكثر صلة)
relevant_docs = collection.query(
    query_embeddings=[query_embedding], # الـ Embedding الخاص بالسؤال
    n_results=3                          # عدد النتائج المراد استرجاعها
)
print_arabic("تم استرجاع المستندات ذات الصلة.")

# تجميع المحتوى من المستندات المسترجعة لإنشاء "السياق"
context = "\n".join(relevant_docs['documents'][0])
print_arabic("\nالسياق المسترجع:")
print_arabic(context)
print_arabic("-" * 50)

# 3. تعزيز السؤال (Augmentation)
print_arabic("جاري تعزيز السؤال بالسياق المسترجع...")
# بناء الـ Prompt الذي سيُرسل إلى النموذج اللغوي، متضمناً السياق والسؤال الأصلي
prompt = f"""أنت مساعد ذكي. أجب على السؤال بناءً على السياق فقط.
إذا لم تتمكن من العثور على الإجابة في السياق المقدم، أجب بـ "المعلومة غير متوفرة في السياق."
السياق:
---
{context}
---
السؤال: {user_query}
الإجابة:"""

print_arabic("تم بناء الـ Prompt المعزز.")
print_arabic("\nالـ Prompt الكامل:")
print_arabic(prompt)
print_arabic("-" * 50)

# 4. التوليد (Generation) للإجابة النهائية
print_arabic("جاري توليد الإجابة بواسطة النموذج اللغوي...")
# تهيئة النموذج اللغوي التوليدي (Gemini 2.0 Flash Lite)
# توليد المحتوى (الإجابة) بناءً على الـ Prompt المعزز
response = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents=prompt
)
print_arabic("تم توليد الإجابة:")
print_arabic(response.text)
