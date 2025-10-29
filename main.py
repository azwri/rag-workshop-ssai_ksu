#%%
from google import genai
from openai import OpenAI
import chromadb
import os
from dotenv import load_dotenv
import print_arabic

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


#%%

user_query = "متى تأسست شركة النور ومن هو رئيسها التنفيذي؟"

print(f"سؤال المستخدم: {user_query}")

print("-" * 100)
print("إجابة بدون RAG: -- GEMINI 2.0 Flash Lite")
response = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents=user_query
)

print(response.text)
print("-" * 100)
print("إجابة بدون RAG: -- OPENAI")
response_openai = client_openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": user_query}
    ]
)

print(response_openai.choices[0].message.content)
print("-" * 100)

#%%

db = chromadb.PersistentClient(path="./chroma_db")

collection = db.get_or_create_collection(name="my_documents_collection")

documents = [
    "شركة النور حققت أرباحاً صافية بلغت 100 مليون دولار في الربع الأول من عام 2024. منتجها الأحدث هو 'النور برو', وهو حل برمجي لإدارة الطاقة.",
    "تأسست شركة النور في عام 2000، وتتخصص في تطوير حلول الطاقة المتجددة المبتكرة. مقرها الرئيسي في دبي.",
    "أرباح الربع الثاني لشركة النور في 2024 بلغت 120 مليون دولار، بزيادة قدرها 20% عن الربع السابق. يتميز 'النور برو' بواجهة مستخدم سهلة ودعم للذكاء الاصطناعي.",
    "الرئيس التنفيذي لشركة النور هو السيد أحمد السلمان، وقد أعلن عن خطط توسع عالمية في مؤتمر صحفي."
]


existing_count = collection.count()

if existing_count > 0:
    print(f"تم العثور على {existing_count} مستند محفوظ مسبقاً في قاعدة البيانات!")
else:
    print("لا توجد مستندات محفوظة. جاري إنشاء Embeddings جديدة...")
    result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=documents)

    print(result.embeddings)
    embeddings = [e.values for e in result.embeddings]

    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    print(f"تم تحويل {len(documents)} مستندات إلى Embeddings وحفظها في قاعدة البيانات!")

print("-" * 100)


#%%

user_query = "متى تأسست شركة النور ومن هو رئيسها التنفيذي؟"

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=user_query,
)
query_embedding = result.embeddings[0].values
print("تم إنشاء Embedding لسؤال المستخدم.")
print(query_embedding)
print("-" * 100)

relevant_docs = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


context = "\n".join(relevant_docs['documents'][0])
print("\nالسياق المسترجع:")
print(context)
print("-" * 100)

#%%

prompt = f"""أنت مساعد ذكي. أجب على السؤال بناءً على السياق فقط.
إذا لم تتمكن من العثور على الإجابة في السياق المقدم، أجب بـ "المعلومة غير متوفرة في السياق."
السياق:
---
{context}
---
السؤال: {user_query}
الإجابة:"""


response = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents=prompt
)
print(response.text)
# %%
