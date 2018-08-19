import tkinter as tk


root = tk.Tk()


menu_main = tk.Menu(root, tearoff=0)
root.config(menu=menu_main)

menu_file = tk.Menu(menu_main, tearoff=0)
menu_main.add_cascade(label="File", menu=menu_file)
for label in ("New", "Open", "Save", "Save As", "Print"):
    menu_file.add_command(label=label)
menu_file.add_separator()
menu_recentfiles = tk.Menu(menu_file, tearoff=0)
menu_file.add_cascade(label="Recent Files", menu=menu_recentfiles)
for index, label in enumerate(("template.doc", "thingo.ppt", "banana.py", "circle.bmp", "green.jpg"), 1):
    menu_recentfiles.add_command(label=f"{index}: {label}")
menu_file.add_command(label="Close")
menu_file.add_separator()
menu_file.add_command(label="Close")

menu_edit = tk.Menu(menu_main, tearoff=0)
menu_main.add_cascade(label="Edit", menu=menu_edit)
for label in ("Undo", "Redo"):
    menu_edit.add_command(label=label)
menu_edit.add_separator()
for label in ("Cut", "Copy", "Paste", "Select All"):
    menu_edit.add_command(label=label)
menu_edit.add_separator()
for label in ("Find", "Replace"):
    menu_edit.add_command(label=label)

menu_help = tk.Menu(menu_main, tearoff=0)
menu_main.add_cascade(label="Help", menu=menu_help)
menu_help.add_command(label="About")


class CounterMenu:
    def __init__(self, master, label):
        self.counter = 1
        self.menu = tk.Menu(master, tearoff=0)
        master.add_cascade(label=label, menu=self.menu)
        self.menu.add_command(label="Print 1", command=self.print)

    def print(self):
        print(self.counter)
        self.counter += 1
        self.menu.entryconfig(0, label=f"Print {self.counter}")


menu_counter = CounterMenu(menu_main, "Counter")

big_canvas = tk.Canvas(root, width=200, height=100)
big_canvas.pack()

big_canvas.create_rectangle(10, 10, 190, 90, fill="lightgreen")
big_canvas.create_oval(30, 20, 90, 80, fill="lightblue")
big_canvas.create_oval(110, 20, 170, 80, fill="lightcoral")

root.mainloop()
