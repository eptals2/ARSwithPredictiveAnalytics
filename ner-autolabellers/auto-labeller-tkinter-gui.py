import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import re
import os
import json

class NERLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tkinter NER Labeler")
        self.root.geometry("900x700")
        
        # Entity types with corresponding colors and keyboard shortcuts
        self.entity_types = {
            "AGE": {"color": "#FF5733", "key": "1", "count": 0},
            "GENDER": {"color": "#33FF57", "key": "2", "count": 0},
            "ADDRESS": {"color": "#3357FF", "key": "3", "count": 0},
            "SOFT_SKILL": {"color": "#FF33A8", "key": "4", "count": 0},
            "HARD_SKILL": {"color": "#33A8FF", "key": "5", "count": 0},
            "EDUCATION_LEVEL": {"color": "#A833FF", "key": "6", "count": 0},
            "COURSE": {"color": "#FFA833", "key": "7", "count": 0},
            "EXPERIENCE": {"color": "#33FFA8", "key": "8", "count": 0},
            "CERTIFICATION": {"color": "#A8FF33", "key": "9", "count": 0}
        }
        
        self.current_entity = None
        self.tokens = []
        self.labeled_tokens = []
        
        # Create the main frame
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the input text area
        self.input_frame = ttk.LabelFrame(self.main_frame, text="Input Text")
        self.input_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.text_input = scrolledtext.ScrolledText(self.input_frame, height=8)
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create buttons frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        self.process_btn = ttk.Button(self.button_frame, text="Process Text", command=self.process_text)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(self.button_frame, text="Clear All", command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Create entity buttons frame
        self.entity_frame = ttk.LabelFrame(self.main_frame, text="Entity Types (Press number key to select)")
        self.entity_frame.pack(fill=tk.X, pady=5)
        
        # Create entity buttons
        self.entity_buttons = {}
        for i, (entity, data) in enumerate(self.entity_types.items()):
            btn = ttk.Button(
                self.entity_frame, 
                text=f"{data['key']}: {entity} (0)",
                command=lambda e=entity: self.select_entity(e)
            )
            btn.grid(row=i//3, column=i%3, padx=5, pady=5, sticky=tk.W)
            self.entity_buttons[entity] = btn
        
        # Create text display area for labeling
        self.display_frame = ttk.LabelFrame(self.main_frame, text="Text for Labeling")
        self.display_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.text_display = tk.Text(self.display_frame, wrap=tk.WORD, height=10)
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.text_display.config(state=tk.DISABLED)
        
        # Create export frame
        self.export_frame = ttk.LabelFrame(self.main_frame, text="Export")
        self.export_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.export_format = ttk.Combobox(self.export_frame, values=["CoNLL Format", "JSON Format"])
        self.export_format.current(0)
        self.export_format.pack(fill=tk.X, padx=5, pady=5)
        
        self.export_btn = ttk.Button(self.export_frame, text="Export", command=self.export_data)
        self.export_btn.pack(padx=5, pady=5)
        
        self.export_text = scrolledtext.ScrolledText(self.export_frame, height=8)
        self.export_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind keyboard shortcuts
        self.root.bind("<Key>", self.key_press)
        
        # Show instructions
        messagebox.showinfo(
            "Instructions", 
            "1. Paste your text and click 'Process Text'\n"
            "2. Press number keys 1-9 to select entity types\n"
            "3. Click on words to label them\n"
            "4. Press Enter to finish labeling\n"
            "5. Press Escape to cancel selection"
        )
    
    def process_text(self):
        """Process the input text into tokens"""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to process.")
            return
        
        # Reset previous data
        self.tokens = []
        self.labeled_tokens = []
        self.reset_entity_counts()
        
        # Simple tokenization (split by whitespace and punctuation)
        raw_tokens = re.findall(r'\S+|\s+', text)
        
        for i, token in enumerate(raw_tokens):
            self.tokens.append({
                "id": i,
                "text": token,
                "entity": None,
                "is_whitespace": token.isspace()
            })
        
        self.render_tokens()
    
    def render_tokens(self):
        """Render tokens in the text display area"""
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete("1.0", tk.END)
        
        # Remove all tags
        for tag in self.text_display.tag_names():
            if tag != "sel":  # Don't remove selection tag
                self.text_display.tag_remove(tag, "1.0", tk.END)
        
        # Add tokens to display
        for token in self.tokens:
            token_text = token["text"]
            
            # Insert the token text
            start_index = self.text_display.index(tk.INSERT)
            self.text_display.insert(tk.END, token_text)
            end_index = self.text_display.index(tk.INSERT)
            
            # If not whitespace, create a tag for this token
            if not token["is_whitespace"]:
                tag_name = f"token_{token['id']}"
                self.text_display.tag_add(tag_name, start_index, end_index)
                
                # Configure tag for clickable behavior
                self.text_display.tag_config(
                    tag_name, 
                    background="#f0f0f0" if not token["entity"] else self.entity_types[token["entity"]]["color"],
                    foreground="black" if not token["entity"] else "white"
                )
                self.text_display.tag_bind(tag_name, "<Button-1>", lambda e, t=token: self.toggle_token_label(t))
        
        self.text_display.config(state=tk.DISABLED)
    
    def toggle_token_label(self, token):
        """Toggle the label of a token when clicked"""
        if not self.current_entity:
            messagebox.showwarning("Warning", "Please select an entity type first.")
            return
        
        # If token already has this entity, remove it
        if token["entity"] == self.current_entity:
            token["entity"] = None
            self.update_entity_count(self.current_entity, -1)
        # If token has a different entity, change it
        elif token["entity"]:
            self.update_entity_count(token["entity"], -1)
            token["entity"] = self.current_entity
            self.update_entity_count(self.current_entity, 1)
        # If token has no entity, add the current one
        else:
            token["entity"] = self.current_entity
            self.update_entity_count(self.current_entity, 1)
        
        self.update_labeled_tokens()
        self.render_tokens()
    
    def select_entity(self, entity):
        """Select an entity type"""
        self.current_entity = entity
        self.update_entity_buttons()
    
    def update_entity_buttons(self):
        """Update the visual state of entity buttons"""
        for entity, btn in self.entity_buttons.items():
            if entity == self.current_entity:
                btn.state(["pressed"])
            else:
                btn.state(["!pressed"])
    
    def update_entity_count(self, entity, change):
        """Update the count of entities"""
        self.entity_types[entity]["count"] += change
        count = self.entity_types[entity]["count"]
        key = self.entity_types[entity]["key"]
        
        # Update button text with count
        self.entity_buttons[entity].config(text=f"{key}: {entity} ({count})")
    
    def reset_entity_counts(self):
        """Reset all entity counts to zero"""
        for entity in self.entity_types:
            self.entity_types[entity]["count"] = 0
            key = self.entity_types[entity]["key"]
            self.entity_buttons[entity].config(text=f"{key}: {entity} (0)")
    
    def update_labeled_tokens(self):
        """Update the list of labeled tokens"""
        self.labeled_tokens = [t for t in self.tokens if t["entity"] and not t["is_whitespace"]]
    
    def clear_all(self):
        """Clear all data"""
        self.text_input.delete("1.0", tk.END)
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete("1.0", tk.END)
        self.text_display.config(state=tk.DISABLED)
        self.export_text.delete("1.0", tk.END)
        
        self.tokens = []
        self.labeled_tokens = []
        self.current_entity = None
        self.reset_entity_counts()
        self.update_entity_buttons()
    
    def export_data(self):
        """Export labeled data"""
        if not self.tokens:
            messagebox.showwarning("Warning", "No data to export.")
            return
        
        format_type = self.export_format.get()
        
        if "CoNLL" in format_type:
            output = self.export_to_conll()
        else:  # JSON Format
            output = self.export_to_json()
        
        self.export_text.delete("1.0", tk.END)
        self.export_text.insert(tk.END, output)
        
        # Ask if user wants to save to file
        if messagebox.askyesno("Save to File", "Do you want to save the exported data to a file?"):
            self.save_to_file(output, "CoNLL" in format_type)
    
    def export_to_conll(self):
        """Export data in CoNLL format"""
        output = ""
        token_id = 1
        
        for token in self.tokens:
            if token["is_whitespace"]:
                continue
            
            # If token has a newline, start a new sentence
            if "\n" in token["text"]:
                output += "\n"
                continue
            
            entity_tag = token["entity"] if token["entity"] else "O"
            output += f"{token_id}\t{token['text']}\t{entity_tag}\n"
            token_id += 1
        
        return output
    
    def export_to_json(self):
        """Export data in JSON format"""
        sentences = []
        current_sentence = {"tokens": [], "entities": []}
        
        entity_start_index = -1
        entity_type = None
        token_index = 0
        
        for token in self.tokens:
            if token["is_whitespace"]:
                continue
            
            # Add token to current sentence
            current_sentence["tokens"].append(token["text"])
            
            # If token has an entity, track it
            if token["entity"]:
                if entity_type != token["entity"]:
                    # If we were tracking an entity, add it to the entities list
                    if entity_type and entity_start_index != -1:
                        current_sentence["entities"].append({
                            "start": entity_start_index,
                            "end": token_index - 1,
                            "type": entity_type
                        })
                    
                    # Start tracking a new entity
                    entity_type = token["entity"]
                    entity_start_index = token_index
            else:
                # If token has no entity but we were tracking one, add it to the entities list
                if entity_type and entity_start_index != -1:
                    current_sentence["entities"].append({
                        "start": entity_start_index,
                        "end": token_index - 1,
                        "type": entity_type
                    })
                    
                    # Reset entity tracking
                    entity_type = None
                    entity_start_index = -1
            
            token_index += 1
            
            # If token ends with a period, question mark, or exclamation point, start a new sentence
            if re.search(r'[.!?]$', token["text"]):
                # Add any remaining entity
                if entity_type and entity_start_index != -1:
                    current_sentence["entities"].append({
                        "start": entity_start_index,
                        "end": token_index - 1,
                        "type": entity_type
                    })
                    
                    # Reset entity tracking
                    entity_type = None
                    entity_start_index = -1
                
                sentences.append(current_sentence)
                current_sentence = {"tokens": [], "entities": []}
                token_index = 0
        
        # Add any remaining sentence
        if current_sentence["tokens"]:
            # Add any remaining entity
            if entity_type and entity_start_index != -1:
                current_sentence["entities"].append({
                    "start": entity_start_index,
                    "end": token_index - 1,
                    "type": entity_type
                })
            
            sentences.append(current_sentence)
        
        return json.dumps({"data": sentences}, indent=2)
    
    def save_to_file(self, content, is_conll):
        """Save the exported data to a file"""
        file_extension = ".conll" if is_conll else ".json"
        file_path = filedialog.asksaveasfilename(
            defaultextension=file_extension,
            filetypes=[
                ("CoNLL files" if is_conll else "JSON files", f"*{file_extension}"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Success", f"Data saved to {file_path}")
    
    def key_press(self, event):
        """Handle keyboard shortcuts"""
        key = event.char
        
        # Number keys 1-9 for entity selection
        if key in "123456789":
            for entity, data in self.entity_types.items():
                if data["key"] == key:
                    self.select_entity(entity)
                    break
        
        # Enter key to finish labeling (just for feedback)
        elif event.keysym == "Return":
            pass
        
        # Escape key to cancel current selection
        elif event.keysym == "Escape":
            self.current_entity = None
            self.update_entity_buttons()

if __name__ == "__main__":
    root = tk.Tk()
    app = NERLabelerApp(root)
    root.mainloop()