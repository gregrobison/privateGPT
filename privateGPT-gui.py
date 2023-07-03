#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time
import tkinter as tk
import pandas as pd

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS


class Application(tk.Frame):
    def __init__(self, master=None, qa=None, args=None):
        super().__init__(master)
        self.master = master
        self.qa = qa
        self.args = args
        self.pack()
        self.create_widgets()

        self.result_output.tag_config('question', foreground='red')
        self.result_output.tag_config('answer', foreground='blue')
        self.result_output.tag_config('source', foreground='green')

    def create_widgets(self):
        self.ask_label = tk.Label(self, text="Enter a query:")
        self.ask_label.pack(side="top")

        self.ask_entry = tk.Entry(self, width=100)
        self.ask_entry.bind('<Return>', self.ask_question)  # Bind 'Enter' key to ask_question
        self.ask_entry.pack(side="top")

        self.ask_button = tk.Button(self, text="ASK", fg="red",
                                    command=self.ask_question)
        self.ask_button.pack(side="top")

        self.result_output = tk.Text(self, width=100)
        self.result_output.pack(side="top")

        self.export_button = tk.Button(self, text="EXPORT", fg="green",
                                       command=self.export_to_excel)
        self.export_button.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="black",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def ask_question(self, event=None):  # Added event parameter to handle 'Enter' key press
        query = self.ask_entry.get()
        if query.strip() == "":
            return

        start = time.time()
        res = self.qa(query)
        answer, docs = res['result'], [] if self.args.hide_source else res['source_documents']
        end = time.time()

        self.result_output.insert('end', "\n> Question:\n", 'question')
        self.result_output.insert('end', query + '\n')
        self.result_output.insert('end', f"\n> Answer (took {round(end - start, 2)} s.):\n", 'answer')
        self.result_output.insert('end', answer + '\n')

        for document in docs:
            self.result_output.insert('end', "\n> " + document.metadata["source"] + ":\n", 'source')
            self.result_output.insert('end', document.page_content + '\n')

    def export_to_excel(self):
        output_text = self.result_output.get('1.0', 'end')
        df = pd.DataFrame([output_text])
        df.to_excel('output.xlsx')


def main():
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks,
                           verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch,
                          callbacks=callbacks, verbose=False)
        case _default:
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)
    root = tk.Tk()
    app = Application(master=root, qa=qa, args=args)
    app.mainloop()


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
