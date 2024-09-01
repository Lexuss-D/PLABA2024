import re
import pandas as pd

def extract_simplified_sentence_or_return_original(text):
    # Use regex to find the sentence that follows the phrases indicating simplification
    match = re.search(r"simplified sentence would be:\s*\"(.*?)\"|even simpler terms:\s*\"(.*?)\"", text)

    # If a match is found, return the simplified sentence
    if match:
        return match.group(1) if match.group(1) else match.group(2)

    # If no match is found, return the original text
    return text

def postprocesing(data):
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's a simplified version:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's a simplified version of the sentence:\"",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's a simplified version of the sentence:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's a simplified version:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Simplified:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("**Simplified**:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's a simplified version of the given sentence:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's the simplified version of the sentence:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's the simplified version:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's the simplified sentence:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's a simplified version of the given text:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's a simplified version of the provided sentence:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's the simplified version of the given sentence:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("The simplified sentence is:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("Here's the simplified text:",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("\"",""))
  data['generated']=data['generated'].apply(lambda x: x.replace("\"",""))
  data['generated']=data['generated'].apply(lambda x: extract_simplified_sentence_or_return_original(x))
  return data

if __name__ == "__main__":
   filename = "/clwork/zhidong/llama/data/plaba_generated_70b_instruct_testdata_postprocessed.tsv"
   plaba = pd.read_csv(filename,index_col=None,sep='\t')
   plaba = postprocesing(plaba)
   plaba.to_csv("/clwork/zhidong/llama/data/plaba_generated_70b_instruct_testdata_postprocessed_1.tsv",sep="\t",encoding="utf-8",index=None)



