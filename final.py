import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt

# ==============================================================================
# PHẦN 1: CÁC CLASS CƠ BẢN & LOGIC TÍNH TOÁN (GIỮ NGUYÊN 100% TỪ CODE CŨ)
# ==============================================================================
class Node:
    def __init__(self, name, x, y):
        self.name = name
        self.x = float(x)
        self.y = float(y)

class Member:
    def __init__(self, name, node_i, node_j):
        self.name = name
        self.node_i = node_i
        self.node_j = node_j

    def get_properties(self):
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        length = math.sqrt(dx**2 + dy**2)
        if length == 0: raise ValueError(f"Thanh {self.name} có chiều dài = 0!")
        cos_a = dx / length
        sin_a = dy / length
        return cos_a, sin_a, length

class TrussSolver:
    def __init__(self):
        self.nodes = {}
        self.members = []
        self.supports = {}
        self.loads = []

    def add_node(self, name, x, y):
        self.nodes[name] = Node(name, x, y)

    def add_member(self, n1_name, n2_name):
        # Kiểm tra trùng lặp
        for m in self.members:
            if {m.node_i.name, m.node_j.name} == {n1_name, n2_name}: return
        if n1_name not in self.nodes or n2_name not in self.nodes:
            raise ValueError("Tên nút không tồn tại!")
        name = f"{n1_name}-{n2_name}"
        self.members.append(Member(name, self.nodes[n1_name], self.nodes[n2_name]))

    def add_support(self, type_sup, name, angle):
        rad = math.radians(float(angle))
        self.supports[name] = {
            'type': type_sup, 'angle': float(angle),
            'c': -math.sin(rad), 's': math.cos(rad) # Giữ logic vector cũ của bạn
        }

    def add_load(self, name, P, angle):
        self.loads.append({'node': name.upper(), 'P': float(P), 'angle': float(angle)})

    def clear_all(self):
        self.nodes = {}
        self.members = []
        self.supports = {}
        self.loads = []

    def solve(self):
        num_nodes = len(self.nodes)
        num_members = len(self.members)
        if num_nodes == 0: return {}, {}

        # 1. Đếm số ẩn
        num_reactions = 0
        for s in self.supports.values():
            num_reactions += 2 if s['type'] == 'pin' else 1
        
        num_equations = 2 * num_nodes
        total_unknowns = num_members + num_reactions

        if total_unknowns < num_equations:
            raise ValueError("Hệ biến hình (Thiếu liên kết)")

        # 2. Xây dựng ma trận
        node_keys = list(self.nodes.keys())
        node_idx_map = {name: i for i, name in enumerate(node_keys)}
        member_idx_map = {m.name: i for i, m in enumerate(self.members)}

        A = np.zeros((num_equations, total_unknowns))
        b = np.zeros(num_equations)

        # Tải trọng
        for load in self.loads:
            if load['node'] not in node_idx_map: continue
            idx = node_idx_map[load['node']]
            rad = math.radians(load['angle'])
            b[2*idx]     -= load['P'] * math.cos(rad)
            b[2*idx + 1] -= load['P'] * math.sin(rad)

        # Thanh
        for m in self.members:
            cx, cy, _ = m.get_properties()
            col = member_idx_map[m.name]
            idx_i = node_idx_map[m.node_i.name]
            idx_j = node_idx_map[m.node_j.name]

            A[2*idx_i, col]     += cx
            A[2*idx_i + 1, col] += cy
            A[2*idx_j, col]     -= cx
            A[2*idx_j + 1, col] -= cy

        # Gối đỡ
        current_reac_idx = 0
        reaction_info = []
        for name, sup in self.supports.items():
            idx_node = node_idx_map[name]
            row_x, row_y = 2 * idx_node, 2 * idx_node + 1
            c, s = sup['c'], sup['s']

            if sup['type'] == 'pin':
                col_n, col_t = num_members + current_reac_idx, num_members + current_reac_idx + 1
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
        x_result, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # 4. Trích xuất kết quả
        member_forces = {m.name: x_result[member_idx_map[m.name]] for m in self.members}
        reaction_forces = {}
        for info in reaction_info:
            name = info['name']
            if info['type'] == 'pin':
                Rn, Rt = x_result[info['idx_n']], x_result[info['idx_t']]
                reaction_forces[name] = (Rn * info['c'] - Rt * info['s'], Rn * info['s'] + Rt * info['c'])
            else:
                Rn = x_result[info['idx']]
                reaction_forces[name] = (Rn * info['c'], Rn * info['s'])

        return member_forces, reaction_forces

# ==============================================================================
# PHẦN 2: GIAO DIỆN STREAMLIT (THAY THẾ TKINTER)
# ==============================================================================

# Cấu hình trang
st.set_page_config(page_title="Phân tích Giàn 2D", layout="wide")

# Khởi tạo Session State để lưu dữ liệu khi web reload
if 'truss' not in st.session_state:
    st.session_state.truss = TrussSolver()
if 'solution' not in st.session_state:
    st.session_state.solution = None

truss = st.session_state.truss # Biến tắt cho gọn

# --- THANH BÊN (SIDEBAR) ĐỂ NHẬP LIỆU ---
with st.sidebar:
    st.header("🛠️ Bảng điều khiển")
    
    # Tab nhập liệu
    tab1, tab2, tab3, tab4 = st.tabs(["Nút", "Thanh", "Gối", "Tải"])
    
    with tab1:
        st.subheader("Thêm Nút (Node)")
        with st.form("add_node_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            name = col1.text_input("Tên", max_chars=5).upper()
            x = col2.number_input("X (m)", value=0.0)
            y = col3.number_input("Y (m)", value=0.0)
            if st.form_submit_button("Thêm Nút"):
                if name:
                    try: 
                        truss.add_node(name, x, y)
                        st.success(f"Đã thêm nút {name}")
                        st.session_state.solution = None # Reset kết quả khi sửa mô hình
                    except Exception as e: st.error(str(e))

    with tab2:
        st.subheader("Thêm Thanh (Member)")
        with st.form("add_member_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            node_list = list(truss.nodes.keys())
            n1 = col1.selectbox("Nút đầu", options=node_list) if node_list else col1.text_input("Nút đầu")
            n2 = col2.selectbox("Nút cuối", options=node_list, index=1 if len(node_list)>1 else 0) if node_list else col2.text_input("Nút cuối")
            
            if st.form_submit_button("Thêm Thanh"):
                try: 
                    truss.add_member(n1, n2)
                    st.success(f"Đã nối {n1}-{n2}")
                    st.session_state.solution = None
                except Exception as e: st.error(str(e))

    with tab3:
        st.subheader("Thêm Gối (Support)")
        with st.form("add_sup_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            s_name = col1.selectbox("Tại nút", options=list(truss.nodes.keys())) if truss.nodes else col1.text_input("Nút")
            s_type = col2.selectbox("Loại", options=["pin", "roller"])
            s_angle = st.number_input("Góc nghiêng (độ)", value=0.0)
            if st.form_submit_button("Đặt Gối"):
                try: 
                    truss.add_support(s_type, s_name, s_angle)
                    st.success(f"Đã đặt gối tại {s_name}")
                    st.session_state.solution = None
                except Exception as e: st.error(str(e))

    with tab4:
        st.subheader("Thêm Tải (Load)")
        with st.form("add_load_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            l_name = col1.selectbox("Tại nút", options=list(truss.nodes.keys())) if truss.nodes else col1.text_input("Nút")
            l_val = col2.number_input("Lực P (kN)", value=100.0)
            l_ang = st.number_input("Góc (độ)", value=270.0, help="270 độ là hướng thẳng xuống dưới")
            if st.form_submit_button("Đặt Tải"):
                try: 
                    truss.add_load(l_name, l_val, l_ang)
                    st.success(f"Đã đặt tải tại {l_name}")
                    st.session_state.solution = None
                except Exception as e: st.error(str(e))

    st.divider()
    if st.button("🗑️ Xóa toàn bộ mô hình", type="primary"):
        truss.clear_all()
        st.session_state.solution = None
        st.rerun()

    if st.button("▶️ CHẠY PHÂN TÍCH (SOLVE)", type="primary"):
        try:
            mem_f, reac_f = truss.solve()
            st.session_state.solution = (mem_f, reac_f)
            st.success("Đã giải xong!")
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")

# --- KHUNG HIỂN THỊ CHÍNH ---
st.title("🏗️ Mô Phỏng Giàn Không Gian 2D")

col_main, col_info = st.columns([3, 1])

# HÀM VẼ MATPLOTLIB (Thay thế Canvas của Tkinter)
def draw_truss_matplotlib(solver, solution=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Vẽ Thanh
    for m in solver.members:
        n1, n2 = m.node_i, m.node_j
        color = 'black'
        linewidth = 2
        
        # Nếu đã giải, tô màu theo nội lực
        if solution:
            forces = solution[0]
            if m.name in forces:
                f = forces[m.name]
                if f > 0.001: color = 'blue' # Kéo
                elif f < -0.001: color = 'red' # Nén
                
                # Hiển thị giá trị nội lực giữa thanh
                mid_x, mid_y = (n1.x + n2.x)/2, (n1.y + n2.y)/2
                ax.text(mid_x, mid_y, f"{f:.1f}", color=color, fontsize=9, fontweight='bold', 
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        ax.plot([n1.x, n2.x], [n1.y, n2.y], color=color, linewidth=linewidth, zorder=1)

    # 2. Vẽ Nút
    for n in solver.nodes.values():
        ax.plot(n.x, n.y, 'o', color='white', markeredgecolor='black', markersize=8, zorder=2)
        ax.text(n.x, n.y + 0.3, n.name, fontsize=10, fontweight='bold', ha='center')

    # 3. Vẽ Gối
    for name, s in solver.supports.items():
        n = solver.nodes[name]
        marker = '^' if s['type'] == 'pin' else 'o'
        ax.plot(n.x, n.y - 0.2, marker=marker, color='gray', markersize=12, zorder=1)

    # 4. Vẽ Tải Trọng (Mũi tên)
    for l in solver.loads:
        n = solver.nodes[l['node']]
        rad = math.radians(l['angle'])
        # Vẽ mũi tên hướng vào nút
        dx = 1.5 * math.cos(rad) # Độ dài mũi tên giả định để vẽ
        dy = 1.5 * math.sin(rad)
        
        # Dùng annotate để vẽ mũi tên đẹp hơn
        ax.annotate("", xy=(n.x, n.y), xytext=(n.x - dx, n.y - dy),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
        ax.text(n.x - dx*1.1, n.y - dy*1.1, f"{l['P']}kN", ha='center')

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Sơ đồ kết cấu & Nội lực')
    return fig

with col_main:
    # Vẽ hình
    if truss.nodes:
        fig = draw_truss_matplotlib(truss, st.session_state.solution)
        st.pyplot(fig)
    else:
        st.info("Chưa có dữ liệu. Hãy thêm Nút và Thanh ở menu bên trái.")

with col_info:
    st.subheader("📝 Kết quả")
    if st.session_state.solution:
        mem_forces, reac_forces = st.session_state.solution
        
        st.write("**Nội lực thanh (kN):**")
        # Tạo bảng nhỏ hiển thị lực
        force_data = []
        for k, v in mem_forces.items():
            state = "Kéo" if v > 0.001 else "Nén" if v < -0.001 else "-"
            force_data.append({"Thanh": k, "Lực": f"{v:.2f}", "Trạng thái": state})
        st.dataframe(force_data, hide_index=True)
        
        st.write("**Phản lực gối (kN):**")
        for k, (rx, ry) in reac_forces.items():
            st.write(f"📍 {k}: Rx={rx:.2f}, Ry={ry:.2f}")
    else:
        st.write("Đang chờ tính toán...")
        
    # Load ví dụ
    if st.button("Tải Ví Dụ Mẫu"):
        truss.clear_all()
        # Ví dụ giàn đơn giản
        truss.add_node("A", 0, 0); truss.add_node("B", 4, 0)
        truss.add_node("C", 8, 0); truss.add_node("D", 4, 3)
        truss.add_member("A", "B"); truss.add_member("B", "C")
        truss.add_member("A", "D"); truss.add_member("B", "D"); truss.add_member("C", "D")
        truss.add_support("pin", "A", 0); truss.add_support("roller", "C", 0)
        truss.add_load("D", 100, 270)
        st.session_state.solution = None
        st.rerun()
