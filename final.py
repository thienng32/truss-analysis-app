import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import math

# ==============================================================================
# PHẦN 1: CÁC CLASS CƠ BẢN (NÚT, THANH) - GIỮ NGUYÊN
# ==============================================================================
class Node:
    """Đại diện cho một Nút (Joint)"""
    def __init__(self, name, x, y):
        self.name = name
        self.x = float(x)
        self.y = float(y)

class Member:
    """Đại diện cho một Thanh (Member/Bar) nối giữa 2 nút"""
    def __init__(self, name, node_i, node_j):
        self.name = name
        self.node_i = node_i  # Nút đầu
        self.node_j = node_j  # Nút cuối

    def get_properties(self):
        """Tính chiều dài và cos, sin góc nghiêng của thanh"""
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        length = math.sqrt(dx**2 + dy**2)
        
        if length == 0:
            raise ValueError(f"Hai nút của thanh {self.name} trùng nhau!")
            
        cos_a = dx / length
        sin_a = dy / length
        return cos_a, sin_a, length

# ==============================================================================
# PHẦN 2: LOGIC TÍNH TOÁN GIÀN (CORE) - GIỮ NGUYÊN
# ==============================================================================
class TrussSolver:
    def __init__(self):
        self.nodes = {}       # Chứa danh sách nút: {'A': Node_A, ...}
        self.members = []     # Chứa danh sách thanh: [Member_1, ...]
        self.supports = {}    # Chứa gối đỡ
        self.loads = []       # Chứa tải trọng

    def add_node(self, name, x, y):
        self.nodes[name] = Node(name, x, y)

    def add_member(self, n1_name, n2_name):
        # Kiểm tra thanh đã tồn tại chưa
        for m in self.members:
            existing_set = {m.node_i.name, m.node_j.name}
            if existing_set == {n1_name, n2_name}:
                return # Đã có rồi thì thôi

        if n1_name not in self.nodes or n2_name not in self.nodes:
            raise ValueError("Tên nút không tồn tại!")

        name = f"{n1_name}-{n2_name}"
        member = Member(name, self.nodes[n1_name], self.nodes[n2_name])
        self.members.append(member)

    def add_support(self, type_sup, name, angle):
        if name not in self.nodes:
            raise ValueError(f"Nút {name} chưa có để đặt gối!")
        
        rad = math.radians(float(angle))
        self.supports[name] = {
            'type': type_sup, 
            'angle': float(angle),
            'c': -math.sin(rad),
            's': math.cos(rad)
        }

    def add_load(self, name, P, angle):
        if name not in self.nodes:
            raise ValueError(f"Nút {name} chưa có để đặt tải!")
        self.loads.append({
            'node': name.upper(),
            'P': float(P),
            'angle': float(angle)
        })

    def remove_node(self, name):
        for m in self.members:
            if m.node_i.name == name or m.node_j.name == name:
                raise ValueError(f"Không thể xóa nút {name} vì đang nối với thanh {m.name}")
        
        if name in self.supports: del self.supports[name]
        self.loads = [l for l in self.loads if l['node'] != name]
        del self.nodes[name]

    def remove_member(self, n1, n2):
        new_list = []
        found = False
        for m in self.members:
            if {m.node_i.name, m.node_j.name} == {n1, n2}:
                found = True
                continue
            new_list.append(m)
        
        if not found: raise ValueError("Không tìm thấy thanh để xóa")
        self.members = new_list

    def remove_support(self, name):
        if name in self.supports: del self.supports[name]

    def remove_load(self, node, P, angle):
        for i, l in enumerate(self.loads):
            if (l['node'] == node and 
                math.isclose(l['P'], float(P)) and 
                math.isclose(l['angle'], float(angle))):
                self.loads.pop(i)
                return
        raise ValueError("Không tìm thấy tải trọng này")

    def clear_all(self):
        self.nodes = {}
        self.members = []
        self.supports = {}
        self.loads = []

    def solve(self):
        num_nodes = len(self.nodes)
        num_members = len(self.members)
        if num_nodes == 0: return {}, {}

        # 1. Đếm số ẩn số
        num_reactions = 0
        for s in self.supports.values():
            if s['type'] == 'pin': num_reactions += 2
            else: num_reactions += 1
        
        num_equations = 2 * num_nodes
        total_unknowns = num_members + num_reactions

        if total_unknowns < num_equations:
            raise ValueError("Hệ biến hình (Thiếu liên kết - Cơ cấu)")

        # 2. Xây dựng ma trận
        node_keys = list(self.nodes.keys())
        node_idx_map = {name: i for i, name in enumerate(node_keys)}
        member_idx_map = {m.name: i for i, m in enumerate(self.members)}

        A = np.zeros((num_equations, total_unknowns))
        b = np.zeros(num_equations)

        # 2a. Tải trọng
        for load in self.loads:
            if load['node'] not in node_idx_map: continue
            idx = node_idx_map[load['node']]
            rad = math.radians(load['angle'])
            b[2*idx]     -= load['P'] * math.cos(rad)
            b[2*idx + 1] -= load['P'] * math.sin(rad)

        # 2b. Thanh
        for m in self.members:
            cx, cy, _ = m.get_properties()
            col = member_idx_map[m.name]
            
            idx_i = node_idx_map[m.node_i.name]
            idx_j = node_idx_map[m.node_j.name]

            A[2*idx_i, col]     += cx
            A[2*idx_i + 1, col] += cy
            A[2*idx_j, col]     -= cx
            A[2*idx_j + 1, col] -= cy

        # 2c. Gối đỡ
        current_reac_idx = 0
        reaction_info = []

        for name, sup in self.supports.items():
            idx_node = node_idx_map[name]
            row_x = 2 * idx_node
            row_y = 2 * idx_node + 1
            c, s = sup['c'], sup['s']

            if sup['type'] == 'pin':
                col_n = num_members + current_reac_idx
                col_t = num_members + current_reac_idx + 1
                
                A[row_x, col_n] += c
                A[row_y, col_n] += s
                A[row_x, col_t] -= s
                A[row_y, col_t] += c
                
                reaction_info.append({'name': name, 'type': 'pin', 'idx_n': col_n, 'idx_t': col_t, 'c': c, 's': s})
                current_reac_idx += 2
            else:
                col = num_members + current_reac_idx
                A[row_x, col] += c
                A[row_y, col] += s
                
                reaction_info.append({'name': name, 'type': 'roller', 'idx': col, 'c': c, 's': s})
                current_reac_idx += 1

        # 3. Giải hệ
        x_result, residuals, rank, s_vals = np.linalg.lstsq(A, b, rcond=None)

        if rank < total_unknowns:
            raise ValueError(f"Hệ biến hình hoặc suy biến (Rank={rank})")

        # 4. Trích xuất kết quả
        member_forces = {}
        for m in self.members:
            val = x_result[member_idx_map[m.name]]
            member_forces[m.name] = val

        reaction_forces = {}
        for info in reaction_info:
            name = info['name']
            if info['type'] == 'pin':
                Rn = x_result[info['idx_n']]
                Rt = x_result[info['idx_t']]
                Rx = Rn * info['c'] - Rt * info['s']
                Ry = Rn * info['s'] + Rt * info['c']
                reaction_forces[name] = (Rx, Ry)
            else:
                Rn = x_result[info['idx']]
                Rx = Rn * info['c']
                Ry = Rn * info['s']
                reaction_forces[name] = (Rx, Ry)

        return member_forces, reaction_forces

# ==============================================================================
# PHẦN 3: GIAO DIỆN (GUI)
# ==============================================================================
class TrussApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phần mềm Phân tích Giàn 2D - Full Interactive")
        try: self.state('zoomed')
        except: self.attributes('-zoomed', True)
        
        self.truss = TrussSolver()
        self.solution = None
        
        # Các biến điều khiển View (Zoom/Pan)
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_pan_enabled = True 

        # --- CÁC BIẾN MỚI CHO TÍNH NĂNG CLICK CHỌN THANH ---
        self.add_bar_mode = False # Trạng thái đang chọn thanh
        self.first_node = None    # Lưu nút đầu tiên vừa bấm

        self._setup_ui()

    def _setup_ui(self):
        # Layout chính: Chia đôi Trái/Phải
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # --- KHUNG TRÁI: VẼ ---
        self.visual_frame = ttk.Frame(main_paned)
        main_paned.add(self.visual_frame, weight=1) 
        self._setup_canvas()

        # --- KHUNG PHẢI: ĐIỀU KHIỂN ---
        self.right_frame = ttk.Frame(main_paned, width=260) 
        main_paned.add(self.right_frame, weight=0)
        
        right_paned = ttk.PanedWindow(self.right_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)

        # Tab nhập liệu
        self.input_notebook = ttk.Notebook(right_paned)
        right_paned.add(self.input_notebook, weight=3)

        self.manual_frame = ttk.Frame(self.input_notebook)
        self.input_notebook.add(self.manual_frame, text="📝 Nhập Thủ Công")
        self._setup_manual_tabs()

        self.script_frame = ttk.Frame(self.input_notebook)
        self.input_notebook.add(self.script_frame, text="💻 Nhập Code (Script)")
        self._setup_script_editor()

        # Console (Kết quả)
        console_container = ttk.Frame(right_paned)
        right_paned.add(console_container, weight=1)
        
        tk.Label(console_container, text="KẾT QUẢ & NHẬT KÝ (CONSOLE)", bg="#333", fg="white", anchor="w").pack(fill=tk.X)
        self.console = scrolledtext.ScrolledText(console_container, bg="black", fg="white", font=("Consolas", 9), height=10, width=30)
        self.console.pack(fill=tk.BOTH, expand=True)
        
        tk.Button(console_container, text="🗑 XÓA TOÀN BỘ DỮ LIỆU", bg="#D32F2F", fg="white", command=self.reset_data).pack(fill=tk.X)

    def _setup_canvas(self):
        self.canvas = tk.Canvas(self.visual_frame, bg="white", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.coord_label = tk.Label(self.visual_frame, text="X: 0.0 | Y: 0.0", bg="white", relief="solid", bd=1)
        self.coord_label.place(x=10, y=10)
        
        # Panel nút Zoom/Pan góc dưới
        ctrl_panel = tk.Frame(self.visual_frame, bg="white")
        ctrl_panel.place(relx=0.98, rely=0.98, anchor="se")
        
        tk.Button(ctrl_panel, text="+", font="Arial 12 bold", width=3, command=lambda: self.do_zoom(None, 1)).pack(pady=2)
        tk.Button(ctrl_panel, text="-", font="Arial 12 bold", width=3, command=lambda: self.do_zoom(None, -1)).pack(pady=2)
        
        self.btn_pan = tk.Button(ctrl_panel, text="✋ Pan", font="Arial 10 bold", width=5, bg="#81C784", command=self.toggle_pan)
        self.btn_pan.pack(pady=5)

        # Gán sự kiện chuột
        self.canvas.bind("<Configure>", lambda e: self.draw_structure())
        
        # --- THAY ĐỔI: GỘP XỬ LÝ CLICK ĐỂ HỖ TRỢ CHỌN ĐIỂM ---
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click) 
        # -----------------------------------------------------
        
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.do_zoom)
        self.canvas.bind("<Button-4>", lambda e: self.do_zoom(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self.do_zoom(e, -1))
        self.canvas.bind("<Motion>", self.show_coords)

    # --- CÁC HÀM XỬ LÝ CLICK CHỌN THANH (MỚI) ---
    def toggle_add_bar_mode(self):
        """Bật tắt chế độ click chuột để chọn thanh"""
        self.add_bar_mode = not self.add_bar_mode
        self.first_node = None # Reset điểm chọn dở
        if self.add_bar_mode:
            self.btn_add_interactive.config(text="❌ HỦY (Đang chọn...)", bg="#FF9800", fg="black")
            self.is_pan_enabled = False # Tắt Pan để dễ click
            self.btn_pan.config(bg="#E0E0E0", relief="sunken")
        else:
            self.btn_add_interactive.config(text="👆 Bật chế độ Click chọn điểm", bg="#E0E0E0", fg="black")
            self.is_pan_enabled = True # Bật lại Pan
            self.btn_pan.config(bg="#81C784", relief="raised")
        self.draw_structure()

    def on_canvas_click(self, e):
        """Hàm điều hướng click chuột: Nếu đang chọn thanh thì tìm điểm, nếu không thì Pan"""
        # Nếu đang ở chế độ thêm thanh
        if self.add_bar_mode:
            clicked_node = self.find_node_at_screen_pos(e.x, e.y)
            
            if clicked_node:
                if self.first_node is None:
                    # Chọn điểm đầu
                    self.first_node = clicked_node
                    self.draw_structure() # Vẽ lại để highlight điểm đầu
                else:
                    # Chọn điểm cuối -> Tạo thanh
                    if clicked_node != self.first_node:
                        try:
                            self.truss.add_member(self.first_node, clicked_node)
                            self.sync_ui() # Cập nhật UI
                            self.first_node = None # Reset để chọn thanh tiếp theo luôn
                            self.draw_structure()
                        except Exception as ex:
                            messagebox.showerror("Lỗi", str(ex))
                    else:
                        # Bấm lại vào chính nó -> Hủy chọn điểm đầu
                        self.first_node = None
                        self.draw_structure()
            return # Đã xử lý click, không làm gì thêm

        # Nếu không ở chế độ thêm thanh thì chạy Pan cũ
        self.start_pan(e)

    def find_node_at_screen_pos(self, sx, sy):
        """Tìm xem click chuột có trúng nút nào không"""
        if not hasattr(self, 'view_params'): return None
        min_x, min_y, scale, margin, h = self.view_params
        
        # Bán kính tìm kiếm (pixel)
        search_radius = 12 
        
        for name, n in self.truss.nodes.items():
            # Tính tọa độ màn hình của nút n
            nx = 50 + (n.x - min_x) * scale + self.pan_x
            ny = h - 50 - (n.y - min_y) * scale + self.pan_y
            
            dist = math.sqrt((sx - nx)**2 + (sy - ny)**2)
            if dist <= search_radius:
                return name
        return None
    # ---------------------------------------------

    def toggle_pan(self):
        self.is_pan_enabled = not self.is_pan_enabled
        if self.is_pan_enabled:
            self.btn_pan.config(bg="#81C784", relief="raised")
            # Tự động tắt chế độ thêm thanh nếu bật Pan
            if self.add_bar_mode: self.toggle_add_bar_mode()
        else:
            self.btn_pan.config(bg="#E0E0E0", relief="sunken")

    def _setup_manual_tabs(self):
        nb = ttk.Notebook(self.manual_frame)
        nb.pack(fill=tk.BOTH, expand=True)
        
        # 1. Tab Nút (Giữ nguyên dùng hàm cũ)
        self.tree_nodes = self._add_manual_subtab(nb, "Nút", [("Tên", 5), ("X", 8), ("Y", 8)], self.handle_add_node, ["", "0", "0"])
        
        # 2. Tab Thanh (SỬA LẠI: Tự tạo Frame để nhét nút bấm vào)
        f_bar = ttk.Frame(nb)
        nb.add(f_bar, text="Thanh")
        
        # --- Nút bấm thêm vào đây ---
        self.btn_add_interactive = tk.Button(f_bar, text="👆 Bật chế độ Click chọn điểm", bg="#E0E0E0", command=self.toggle_add_bar_mode)
        self.btn_add_interactive.pack(fill=tk.X, padx=2, pady=2)
        
        # Gọi hàm phụ trợ mới để vẽ phần nhập liệu + bảng bên dưới nút bấm
        self.tree_bars = self._add_manual_subtab_content(f_bar, "Thanh", [("Đầu", 8), ("Cuối", 8)], self.handle_add_bar, ["", ""])
        
        # 3. Các tab còn lại (Giữ nguyên)
        self.tree_sups = self._add_manual_subtab(nb, "Gối", [("Tên", 5), ("Loại", 8), ("Góc", 5)], self.handle_add_sup, ["", "pin", "0"])
        self.tree_loads = self._add_manual_subtab(nb, "Tải", [("Tên", 5), ("P", 8), ("Góc", 5)], self.handle_add_load, ["", "100", "270"])
        
        tk.Button(self.manual_frame, text="▶ GIẢI HỆ (SOLVE)", bg="#007ACC", fg="white", font="Arial 10 bold", command=self.solve_truss).pack(fill=tk.X, pady=5)

    def _add_manual_subtab(self, nb, title, fields, cmd_func, defaults):
        f = ttk.Frame(nb)
        nb.add(f, text=title)
        
        # Tái sử dụng hàm content bên dưới
        return self._add_manual_subtab_content(f, title, fields, cmd_func, defaults)
    
    # --- HÀM PHỤ TRỢ MỚI ---
    def _add_manual_subtab_content(self, parent, title, fields, cmd_func, defaults):
        """Hàm phụ trợ: Vẽ nội dung (Entry + Table) vào trong một Frame có sẵn"""
        inp_frame = ttk.Frame(parent)
        inp_frame.pack(pady=5, fill=tk.X)
        
        entries = []
        for i, (lbl, w) in enumerate(fields):
            tk.Label(inp_frame, text=lbl).grid(row=0, column=i)
            e = ttk.Entry(inp_frame, width=w)
            e.grid(row=1, column=i, padx=2)
            if i < len(defaults): e.insert(0, defaults[i])
            entries.append(e)
            
        ttk.Button(inp_frame, text="+", width=3, command=lambda: cmd_func(entries)).grid(row=1, column=len(fields))
        
        # Tạo bảng và Căn giữa chữ
        tree = ttk.Treeview(parent, columns=[x[0] for x in fields], show="headings", height=6)
        for x in fields: 
            tree.heading(x[0], text=x[0])
            tree.column(x[0], width=40, anchor="center") # Đã thêm căn giữa
        tree.pack(fill=tk.BOTH, expand=True)
        
        type_map = {"Nút": "node", "Thanh": "bar", "Gối": "sup", "Tải": "load"}
        ttk.Button(parent, text="Xóa dòng chọn", command=lambda: self.del_selected(tree, type_map[title])).pack(fill=tk.X)
        return tree
    # -----------------------

    def _setup_script_editor(self):
        tool_fr = tk.Frame(self.script_frame, bg="#333")
        tool_fr.pack(fill=tk.X)
        tk.Button(tool_fr, text="▶ CHẠY", bg="#007ACC", fg="white", command=self.run_script).pack(side=tk.LEFT, padx=3, pady=2)
        tk.Button(tool_fr, text="VÍ DỤ", bg="#555", fg="white", command=self.load_example).pack(side=tk.LEFT, padx=3, pady=2)
        
        self.editor = scrolledtext.ScrolledText(self.script_frame, bg="#1E1E1E", fg="#00FF00", insertbackground="white", font=("Consolas", 11), width=30)
        self.editor.pack(fill=tk.BOTH, expand=True)
        self.load_example()

    def log(self, text, color="white"):
        self.console.config(state='normal')
        self.console.insert(tk.END, text + "\n", str(color))
        self.console.tag_config(str(color), foreground=color)
        self.console.see(tk.END)
        self.console.config(state='disabled')

    # --- CÁC HÀM XỬ LÝ SỰ KIỆN (HANDLERS) ---
    def handle_add_node(self, entries):
        try: 
            self.truss.add_node(entries[0].get().upper(), entries[1].get(), entries[2].get())
            self.sync_ui()
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def handle_add_bar(self, entries):
        try: 
            self.truss.add_member(entries[0].get().upper(), entries[1].get().upper())
            self.sync_ui()
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def handle_add_sup(self, entries):
        try: 
            self.truss.add_support(entries[1].get().lower(), entries[0].get().upper(), entries[2].get())
            self.sync_ui()
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def handle_add_load(self, entries):
        try: 
            self.truss.add_load(entries[0].get().upper(), entries[1].get(), entries[2].get())
            self.sync_ui()
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def del_selected(self, tree, type_tag):
        sel = tree.selection()
        if not sel: return
        val = tree.item(sel)['values']
        try:
            if type_tag == 'node': self.truss.remove_node(str(val[0]))
            elif type_tag == 'bar': self.truss.remove_member(str(val[0]), str(val[1]))
            elif type_tag == 'sup': self.truss.remove_support(str(val[0]))
            elif type_tag == 'load': self.truss.remove_load(str(val[0]), val[1], val[2])
            self.sync_ui()
        except Exception as e: messagebox.showerror("Lỗi", str(e))

    def run_script(self):
        self.console.config(state='normal')
        self.console.delete('1.0', tk.END)
        self.console.config(state='disabled')
        
        self.truss.clear_all()
        self.solution = None
        lines = self.editor.get("1.0", tk.END).split('\n')
        
        self.log(">>> BẮT ĐẦU CHẠY SCRIPT...", "cyan")
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"): continue
            
            cmd = parts[0].upper()
            try:
                if cmd == "NODE": 
                    self.truss.add_node(parts[1].upper(), parts[2], parts[3])
                    self.log(f"Add Node {parts[1]}", "gray")
                elif cmd in ["BAR", "MEMBER"]:
                    if len(parts) == 2: n1, n2 = parts[1][0], parts[1][1]
                    else: n1, n2 = parts[1], parts[2]
                    self.truss.add_member(n1.upper(), n2.upper())
                    self.log(f"Add Bar {n1}-{n2}", "gray")
                elif cmd in ["PIN", "ROLLER"]:
                    ang = parts[2] if len(parts) > 2 else 0
                    self.truss.add_support(cmd.lower(), parts[1].upper(), ang)
                    self.log(f"Set {cmd} at {parts[1]}", "gray")
                elif cmd == "LOAD":
                    self.truss.add_load(parts[1].upper(), parts[2], parts[3])
                    self.log(f"Add Load at {parts[1]}", "gray")
                elif cmd == "SOLVE":
                    self.solve_truss()
            except Exception as e:
                self.log(f"❌ Line {i+1}: {e}", "red")
        self.sync_ui()

    def load_example(self):
        ex = "# Ví dụ giàn \nNODE A 0 0\nNODE B 4 0 \nNODE C 8 0 \nNODE D 12 0\nNODE E 6 6\nBAR AB\nBAR BC\nBAR CD\nBAR AE\nBAR BE\nBAR CE\nBAR DE\nPIN A 0\nROLLER D 0\nLOAD B 3 270\nLOAD C 6 270\nSOLVE"
        self.editor.delete('1.0', tk.END)
        self.editor.insert(tk.END, ex)

    def solve_truss(self):
        try:
            mem_forces, reac_forces = self.truss.solve()
            self.solution = mem_forces
            
            self.log("=== KẾT QUẢ NỘI LỰC THANH ===", "#00FF00")
            for k, v in mem_forces.items():
                tag = "KÉO" if v > 0.001 else "NÉN" if v < -0.001 else "-"
                self.log(f"{k}: {v:.2f} ({tag})", "white")
            
            self.log("=== PHẢN LỰC GỐI (kN) ===", "#00FFFF")
            for node, (rx, ry) in reac_forces.items():
                self.log(f"Nút {node}: Rx={rx:.2f}, Ry={ry:.2f}", "white")
                
            self.draw_structure()
        except Exception as e: self.log(f"❌ Lỗi giải: {e}", "red")

    def sync_ui(self):
        for t in [self.tree_nodes, self.tree_bars, self.tree_sups, self.tree_loads]:
            for x in t.get_children(): t.delete(x)
            
        for n in self.truss.nodes.values(): 
            self.tree_nodes.insert("", "end", values=(n.name, n.x, n.y))
        for m in self.truss.members: 
            self.tree_bars.insert("", "end", values=(m.node_i.name, m.node_j.name))
        for k, v in self.truss.supports.items(): 
            self.tree_sups.insert("", "end", values=(k, v['type'], v['angle']))
        for l in self.truss.loads: 
            self.tree_loads.insert("", "end", values=(l['node'], l['P'], l['angle']))
            
        self.draw_structure()

    def reset_data(self):
        self.truss.clear_all()
        self.solution = None
        self.console.config(state='normal')
        self.console.delete('1.0', tk.END)
        self.console.config(state='disabled')
        self.sync_ui()

    # --- CÁC HÀM VẼ (VISUALIZATION) MỚI ĐƯỢC CẬP NHẬT ---
    def get_grid_step(self, scale):
        """Tính bước nhảy của lưới tọa độ dựa trên tỷ lệ zoom"""
        raw_step = 80 / scale if scale > 0 else 1 # Khoảng cách lưới mong muốn là ~80px
        
        # Làm tròn về các số đẹp: 1, 2, 5
        exponent = math.floor(math.log10(raw_step))
        base = raw_step / (10**exponent)
        
        if base < 1.5: step = 1
        elif base < 3.5: step = 2
        elif base < 7.5: step = 5
        else: step = 10
        
        return step * (10**exponent)

    def draw_structure(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        # 1. Tính toán vùng bao (Bounding Box)
        if not self.truss.nodes:
            min_x, min_y, max_x, max_y = -5, -5, 5, 5
        else:
            xs = [n.x for n in self.truss.nodes.values()]
            ys = [n.y for n in self.truss.nodes.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Nếu chỉ có 1 điểm hoặc các điểm thẳng hàng, tạo vùng đệm
            if min_x == max_x: min_x -= 5; max_x += 5
            if min_y == max_y: min_y -= 5; max_y += 5

        # 2. Tính tỷ lệ scale & Lưu tham số
        # scale = pixels / đơn vị thực tế
        scale_x = (w - 100) / (max_x - min_x)
        scale_y = (h - 100) / (max_y - min_y)
        base_scale = min(scale_x, scale_y)
        
        current_scale = base_scale * self.zoom_scale
        
        # Lưu view params để dùng cho sự kiện chuột (Zoom/Pan/Coords)
        # (min_x, min_y, scale, margin, height)
        self.view_params = (min_x, min_y, current_scale, 50, h)

        # Hàm nội bộ để chuyển tọa độ thế giới -> màn hình
        def to_screen(x, y):
            scr_x = 50 + (x - min_x) * current_scale + self.pan_x
            scr_y = h - 50 - (y - min_y) * current_scale + self.pan_y
            return scr_x, scr_y

        # 3. Vẽ Lưới (Grid) Thông Minh & Trục tọa độ
        grid_step = self.get_grid_step(current_scale)
        
        # Tính vùng hiển thị thực tế trên màn hình
        # Visible World X min/max
        visible_min_x = (0 - self.pan_x - 50) / current_scale + min_x
        visible_max_x = (w - self.pan_x - 50) / current_scale + min_x
        # Visible World Y min/max (Do trục Y ngược)
        visible_min_y = (h - 50 + self.pan_y - h) / current_scale + min_y
        visible_max_y = (h - 50 + self.pan_y) / current_scale + min_y
        
        origin_x, origin_y = to_screen(0, 0)

        # Vẽ các đường dọc (Grid dọc)
        start_i = math.floor(visible_min_x / grid_step)
        end_i = math.ceil(visible_max_x / grid_step)
        
        for i in range(start_i, end_i + 1):
            val = i * grid_step
            sx, _ = to_screen(val, 0)
            
            # Vẽ đường lưới mờ
            self.canvas.create_line(sx, 0, sx, h, fill="#EEE")
            
            # Vẽ số tọa độ (tránh đè lên trục chính)
            if not math.isclose(val, 0):
                self.canvas.create_text(sx, origin_y + 12, text=f"{float(f'{val:.5f}'):g}", fill="#888", font="Arial 8")

        # Vẽ các đường ngang (Grid ngang)
        start_j = math.floor(visible_min_y / grid_step)
        end_j = math.ceil(visible_max_y / grid_step)
        
        for j in range(start_j, end_j + 1):
            val = j * grid_step
            _, sy = to_screen(0, val)
            
            # Vẽ đường lưới mờ
            self.canvas.create_line(0, sy, w, sy, fill="#EEE")
            
            # Vẽ số tọa độ
            if not math.isclose(val, 0):
                self.canvas.create_text(origin_x - 15, sy, text=f"{float(f'{val:.5f}'):g}", fill="#888", font="Arial 8", anchor="e")

        # 4. Vẽ trục Oxy (Đậm hơn lưới một chút, màu xám)
        self.canvas.create_line(0, origin_y, w, origin_y, fill="#C0C0C0", width=1) # Trục X
        self.canvas.create_line(origin_x, 0, origin_x, h, fill="#C0C0C0", width=1) # Trục Y
        self.canvas.create_text(origin_x - 10, origin_y + 12, text="O", font="Arial 9 bold", fill="#888")

        # 5. Vẽ Thanh (Members)
        for m in self.truss.members:
            x1, y1 = to_screen(m.node_i.x, m.node_i.y)
            x2, y2 = to_screen(m.node_j.x, m.node_j.y)
            
            color = "black"
            width = 2
            
            # Tô màu kết quả nội lực
            if self.solution and m.name in self.solution:
                force = self.solution[m.name]
                if force > 0.001: color = "blue"; width = 4
                elif force < -0.001: color = "red"; width = 4
            
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
            
            # --- ĐOẠN SỬA ĐỔI ĐỂ HIỆN GIÁ TRỊ LỰC ---
            mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
            
            # Kiểm tra: Nếu đã có kết quả giải -> Hiện số + kN, Nếu chưa -> Hiện tên
            if self.solution and m.name in self.solution:
                text_str = f"{self.solution[m.name]:.2f} kN"
                w_box = 30 # Hộp rộng hơn để chứa đủ số
            else:
                text_str = m.name
                w_box = 12 # Hộp nhỏ vừa tên
            
            # Vẽ hộp nền trắng
            self.canvas.create_rectangle(mid_x-w_box, mid_y-8, mid_x+w_box, mid_y+8, fill="white", outline="")
            # Vẽ chữ (text_str đã xác định ở trên)
            self.canvas.create_text(mid_x, mid_y, text=text_str, font="Arial 8 bold", fill="black")

        # 6. Vẽ Gối đỡ (Supports) - (ĐÃ SỬA: Xoay theo góc nhập)
        for name, s in self.truss.supports.items():
            nx, ny = to_screen(self.truss.nodes[name].x, self.truss.nodes[name].y)
            
            # --- TÍNH TOÁN GÓC XOAY ---
            # Dùng dấu âm (-s['angle']) để xoay đúng chiều kim đồng hồ thực tế
            # vì trục Y của màn hình máy tính hướng xuống dưới
            rad = math.radians(-s['angle'])
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)

            # Hàm con: Nhập tọa độ cục bộ (dx, dy), trả về tọa độ màn hình đã xoay
            def get_rot_pos(dx, dy):
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                return nx + rx, ny + ry

            # Kích thước
            sz = 12   # Bán kính ngang
            h_tri = 18 # Chiều cao

            # --- VẼ TAM GIÁC ---
            # 3 điểm: Đỉnh (0,0), Góc trái dưới (-sz, h), Góc phải dưới (sz, h)
            p_top = (nx, ny)
            p_left = get_rot_pos(-sz, h_tri)
            p_right = get_rot_pos(sz, h_tri)

            self.canvas.create_polygon(p_top[0], p_top[1], 
                                       p_left[0], p_left[1], 
                                       p_right[0], p_right[1], 
                                       fill="white", outline="black", width=2)

            # --- VẼ BÁNH XE (Chỉ dành cho Roller) ---
            if s['type'] == 'roller':
                r = 3.5
                # Tâm bánh xe nằm dưới đáy tam giác một đoạn r
                y_wheel_local = h_tri + r 
                
                # Vẽ 3 bánh xe xoay theo tam giác
                for offset in [-8, 0, 8]:
                    # Tính tâm bánh xe mới sau khi xoay
                    cx, cy = get_rot_pos(offset, y_wheel_local)
                    self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, 
                                            fill="white", outline="black", width=2)

        # 7. Vẽ Tải trọng (Loads)
        for l in self.truss.loads:
            nx, ny = to_screen(self.truss.nodes[l['node']].x, self.truss.nodes[l['node']].y)
            rad = math.radians(l['angle'])
            
            arrow_len = 60
            dx = arrow_len * math.cos(rad)
            dy = -arrow_len * math.sin(rad)
            
            # Giữ lại Mũi tên 1 (có arrowshape), bỏ mũi tên 2
            # Tôi để width=4 cho thân mũi tên cân đối với cái đầu to
            self.canvas.create_line(nx-dx, ny-dy, nx, ny, arrow=tk.LAST, 
                                    fill="black", width=4, 
                                    arrowshape=(25, 30, 10))
            
            self.canvas.create_text(nx-dx*1.2, ny-dy*1.2, text=f"{l['P']} kN", fill="black", font="Arial 12 bold")

        # 8. Vẽ Nút (Nodes)
        for n in self.truss.nodes.values():
            nx, ny = to_screen(n.x, n.y)
            
            # Mặc định trắng, viền đen
            fill_color = "white"
            outline_color = "black"
            radius = 4
            
            # --- LOGIC ĐỔI MÀU KHI ĐANG CHỌN ĐIỂM ---
            if self.add_bar_mode and n.name == self.first_node:
                fill_color = "#FFD700" # Vàng gold
                outline_color = "red"
                radius = 6 # Vẽ to hơn chút
            # ----------------------------------------
            
            self.canvas.create_oval(nx-radius, ny-radius, nx+radius, ny+radius, fill=fill_color, outline=outline_color)
            self.canvas.create_text(nx, ny-15, text=n.name, font="Arial 9 bold")

    # --- LOGIC ZOOM / PAN / MOUSE ---
    def start_pan(self, e):
        if not self.is_pan_enabled: return
        self.drag_start_x = e.x
        self.drag_start_y = e.y

    def do_pan(self, e):
        if not self.is_pan_enabled: return
        self.pan_x += e.x - self.drag_start_x
        self.pan_y += e.y - self.drag_start_y
        self.drag_start_x = e.x
        self.drag_start_y = e.y
        self.draw_structure()

    def do_zoom(self, e, direction=None):
        """Xử lý zoom thông minh: Zoom tại chuột nếu đang Pan, Zoom tại tâm nếu không"""
        # Xác định chiều zoom
        if direction:
            delta = direction
        else:
            delta = 1 if e.delta > 0 else -1
            
        zoom_factor = 1.2 if delta > 0 else 0.8

        if not hasattr(self, 'view_params'): return
        min_x, min_y, current_scale, margin, h = self.view_params

        # Xác định tâm Zoom
        if self.is_pan_enabled and e: 
            # Nếu đang bật chế độ Pan và dùng chuột lăn -> Zoom tại vị trí chuột
            center_x = e.x
            center_y = e.y
        else:
            # Nếu tắt Pan hoặc dùng nút bấm -> Zoom tại tâm màn hình
            center_x = self.canvas.winfo_width() / 2
            center_y = self.canvas.winfo_height() / 2

        # Tính tọa độ thực tế (World Coord) tại tâm zoom hiện tại
        world_x = (center_x - margin - self.pan_x) / current_scale + min_x
        world_y = (h - margin + self.pan_y - center_y) / current_scale + min_y

        # Cập nhật scale mới
        self.zoom_scale *= zoom_factor
        new_scale = current_scale * zoom_factor

        # Tính lại Pan để giữ điểm world_x, world_y nằm đúng tại center_x, center_y
        self.pan_x = center_x - (margin + (world_x - min_x) * new_scale)
        self.pan_y = center_y - (h - margin - (world_y - min_y) * new_scale)
        
        self.draw_structure()

    def show_coords(self, e):
        if not hasattr(self, 'view_params'): return
        min_x, min_y, scale, margin, h = self.view_params
        
        # Tính ngược từ màn hình ra tọa độ thực
        wx = (e.x - margin - self.pan_x) / scale + min_x
        wy = (h - margin + self.pan_y - e.y) / scale + min_y
        self.coord_label.config(text=f"X: {wx:.2f} | Y: {wy:.2f}")

if __name__ == "__main__":
    app = TrussApp()
    app.mainloop()