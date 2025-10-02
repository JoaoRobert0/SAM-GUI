import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2


class VisualizadorImagens:
    def __init__(self, master):
        self.master = master
        master.title("Visualizador com SAM")

        # Frame esquerdo: lista de imagens
        self.frame_lista = tk.Frame(master, width=200)
        self.frame_lista.pack(side=tk.LEFT, fill=tk.Y)

        self.botao_pasta = tk.Button(self.frame_lista, text="Selecionar Pasta", command=self.carregar_pasta)
        self.botao_pasta.pack(pady=10)

        self.lista = tk.Listbox(self.frame_lista, width=20)
        self.lista.pack(fill=tk.BOTH, expand=True)
        self.lista.bind("<<ListboxSelect>>", self.exibir_imagem)

        # Frame meio: Canvas para imagem e desenho
        self.frame_meio = tk.Frame(master)
        self.frame_meio.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame_meio, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Frame direito: botões e opções
        self.frame_direito = tk.Frame(master, width=200)
        self.frame_direito.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.botao_sam = tk.Button(self.frame_direito, text="Executar SAM", command=self.executar_sam)
        self.botao_sam.pack(pady=10)

        self.botao_limpar = tk.Button(self.frame_direito, text="Limpar Retângulos", command=self.limpar_retangulos)
        self.botao_limpar.pack(pady=10)

        self.label_classe = tk.Label(self.frame_direito, text="Classe:")
        self.combo_classe = ttk.Combobox(self.frame_direito, values=["classe1", "classe2", "classe3"])
        self.botao_confirmar = tk.Button(self.frame_direito, text="Confirmar Classe", command=self.confirmar_classe)

        # Armazenamento
        self.imagens = []
        self.imagem_atual = None
        self.imagem_tk = None

        # Retângulos
        self.start_x = None
        self.start_y = None
        self.retangulo_temp = None
        self.retangulos = []

        # Imagem gerada pelo SAM (placeholder)
        self.imagem_sam = None

        # Inicializa o SAM aqui (use o modelo ViT-H e checkpoint local ou baixe antes)
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"  # ajuste o caminho se necessário
        model_type = "vit_h"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

        # Eventos de desenho
        self.canvas.bind("<Button-1>", self.iniciar_desenho)
        self.canvas.bind("<B1-Motion>", self.atualizar_desenho)
        self.canvas.bind("<ButtonRelease-1>", self.finalizar_desenho)

    def carregar_pasta(self):
        pasta = filedialog.askdirectory()
        if not pasta:
            return

        self.lista.delete(0, tk.END)
        self.imagens = []

        extensoes_validas = (".png", ".jpg", ".jpeg", ".bmp")
        for nome_arquivo in os.listdir(pasta):
            if nome_arquivo.lower().endswith(extensoes_validas):
                caminho = os.path.join(pasta, nome_arquivo)
                self.imagens.append(caminho)
                self.lista.insert(tk.END, nome_arquivo)
                
    def exibir_imagem(self, event):
        if not self.imagens:
            return
        sel = self.lista.curselection()
        if not sel:
            return

        caminho = self.imagens[sel[0]]
        imagem = Image.open(caminho)
        imagem.thumbnail((800, 600))

        self.imagem_atual = imagem
        self.imagem_tk = ImageTk.PhotoImage(imagem)

        # Limpa a imagem segmentada ao trocar de imagem
        self.imagem_sam = None  

        # Limpa o canvas e retângulos
        self.canvas.delete("all")
        self.retangulos.clear()

        # Oculta os widgets de seleção de classe
        self.label_classe.pack_forget()
        self.combo_classe.pack_forget()
        self.botao_confirmar.pack_forget()

        # Desenha a imagem original no canvas
        self.canvas.config(width=self.imagem_tk.width(), height=self.imagem_tk.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imagem_tk)

    def _desenhar_imagem_sam(self):
        """Desenha a imagem segmentada sobre a imagem base."""
        sam_tk = ImageTk.PhotoImage(self.imagem_sam)
        self.canvas.image_sam_tk = sam_tk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=sam_tk)

    def iniciar_desenho(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.retangulo_temp = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2
        )

    def atualizar_desenho(self, event):
        if self.retangulo_temp:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.retangulo_temp, self.start_x, self.start_y, cur_x, cur_y)

    def finalizar_desenho(self, event):
        if self.retangulo_temp:
            self.retangulos.append(self.retangulo_temp)
            self.retangulo_temp = None

    def limpar_retangulos(self):
        for ret_id in self.retangulos:
            self.canvas.delete(ret_id)
        self.retangulos.clear()
        print("Retângulos removidos.")

    def executar_sam(self):
        if self.imagem_atual is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem selecionada.")
            return

        # Converte imagem PIL para numpy (RGB)
        imagem_np = np.array(self.imagem_atual.convert("RGB"))

        # Configura o predictor com a imagem
        self.predictor.set_image(imagem_np)

        altura, largura = imagem_np.shape[:2]

        # Define o ponto central para segmentação
        ponto_central = np.array([[largura // 2, altura // 2]])
        label_ponto = np.array([1])  # 1 = ponto positivo (foreground)

        # Executa predição para esse ponto (uma máscara só)
        masks, scores, logits = self.predictor.predict(
            point_coords=ponto_central,
            point_labels=label_ponto,
            multimask_output=False
        )

        mask = masks[0]  # máscara binária (bool)

        # Cria imagem RGBA para sobrepor máscara na imagem original
        imagem_com_mask = cv2.cvtColor(imagem_np, cv2.COLOR_RGB2RGBA)

        # Define a máscara vermelha semi-transparente (alpha = 128)
        imagem_com_mask[mask] = (0, 0, 0, 255)

        # Converte para PIL para exibir no Tkinter
        self.imagem_sam = Image.fromarray(imagem_com_mask)

        # Atualiza o canvas
        self.canvas.delete("all")
        self.imagem_tk = ImageTk.PhotoImage(self.imagem_sam)
        self.canvas.config(width=self.imagem_tk.width(), height=self.imagem_tk.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imagem_tk)

        # Exibe os widgets para confirmar classe
        self.label_classe.pack(pady=5)
        self.combo_classe.pack(pady=5)
        self.botao_confirmar.pack(pady=10)

    def confirmar_classe(self):
        classe_escolhida = self.combo_classe.get()
        if not classe_escolhida:
            messagebox.showwarning("Aviso", "Selecione uma classe.")
            return

        # Atualiza imagem segmentada no canvas
        self.canvas.delete("all")
        self.imagem_tk = ImageTk.PhotoImage(self.imagem_sam)
        self.canvas.config(width=self.imagem_tk.width(), height=self.imagem_tk.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imagem_tk)

        # Esconde widgets após confirmação
        self.label_classe.pack_forget()
        self.combo_classe.pack_forget()
        self.botao_confirmar.pack_forget()

        # Marca como (ok) no Listbox
        indice = self.lista.curselection()
        if indice:
            index = indice[0]
            nome_arquivo = self.lista.get(index)

            # Remove "(ok)" anterior se já houver
            if nome_arquivo.endswith(" (ok)"):
                nome_arquivo = nome_arquivo[:-5]

            nome_arquivo_ok = f"{nome_arquivo} (ok)"
            self.lista.delete(index)
            self.lista.insert(index, nome_arquivo_ok)
            self.lista.selection_set(index)  # Mantém o item selecionado

        messagebox.showinfo("Feito", f"Imagem marcada com classe '{classe_escolhida}'.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualizadorImagens(root)
    root.geometry("1200x700")
    root.mainloop()
