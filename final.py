import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt

# ==============================================================================
# PHẦN 1: CORE LOGIC (GIỮ NGUYÊN 100% CỦA BẠN)
# ==============================================================================
class Node:
    def __init__(self, name, x, y):
        self.name = name; self.x = float(x); self.y = float(y)

class Member:
    def __init__(self, name, node_i, node_j):
        self.name = name; self.node_i = node_i; self.node_j = node_j
    def get_properties(self):
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        length = math.sqrt(dx**2 + dy**2)
        if length == 0: raise ValueError(f"Thanh {self.name} dài = 0!")
        return dx/length, dy/length, length

class TrussSolver:
    def __init__(self):
        self.nodes = {}; self.members = []; self.supports = {}; self.loads = []
    
    def add_node(self, name, x, y): self.nodes[name] = Node(name, x, y)
    
    def add_member(self, n1, n2):
        for m in self.members:
            if {m.node_i.name, m.node_j.name} == {n1, n2}: return
        if n1 not in self.nodes or n2 not in self.nodes: raise ValueError("Thiếu nút!")
        self.members.append(Member(f"{n1}-{n2}", self.nodes[n1], self.nodes[n2]))

    def add_support(self, type_sup, name, angle):
        rad = math.radians(float(angle))
        self.supports[name] = {'type': type_sup, 'angle': float(angle), 'c': -math.sin(rad), 's': math.cos(rad)}

    def add_load(self, name, P, angle):
        self.loads.append({'node': name.upper(), 'P': float(P), 'angle': float(angle)})

    def clear_all(self):
        self.nodes = {}; self.members = []; self.supports = {}; self.loads = []

    def solve(self):
        num_nodes = len(self.nodes); num_members = len(self.members)
        if num_nodes == 0: return {}, {}
        
        num_reactions = sum(2 if s['type']=='pin' else 1 for s in self.supports.values())
        if num_members + num_reactions < 2 * num_nodes: raise ValueError("Hệ biến hình!")

        node_keys = list(self.nodes.keys())
        node_idx = {name: i for i, name in enumerate(node_keys)}
        
        A = np.zeros((2*num_nodes, num_members + num_reactions))
        b = np.zeros(2*num_nodes)

        for l in self.loads:
            if l['node'] in node_idx:
                idx = node_idx[l['node']]; rad = math.radians(l['angle'])
                b[2*idx] -= l['P']*math.cos(rad); b[2*idx+1] -= l['P']*math.sin(rad)

        for i, m in enumerate(self.members):
            c, s, _ = m.get_properties()
            idx1, idx2 = node_idx[m.node_i.name], node_idx[m.node_j.name]
            A[2*idx1, i] += c; A[2*idx1+1, i] += s
            A[2*idx2, i] -= c; A[2*idx2+1, i] -= s

        c_idx = num_members
        reac_map = []
        for name, s in self.supports.items():
            idx = node_idx[name]
            if s['type'] == 'pin':
                A[2*idx, c_idx] += s['c']; A[2*idx+1, c_idx] += s['s']
                A[2*idx, c_idx+1] -= s['s']; A[2*idx+1, c_idx+1] += s['c']
                reac_map.extend([(name, c_idx, 'n'), (name, c_idx+1, 't')])
                c_idx += 2
            else:
                A[2*idx, c_idx] += s['c']; A[2*idx+1, c_idx] += s['s']
                reac_map.append((name, c_idx, 'n'))
                c_idx += 1

        res = np.linalg.lstsq(A, b, rcond=None)[0]
        
        mem_res = {m.name: res[i] for i, m in enumerate(self.members)}
        reac_res = {}
        for name, idx, type_ in reac_map:
            val = res[idx]
            sup = self.supports[name]
            if name not in reac_res: reac_res[name] = [0, 0]
            if type_ == 'n':
                reac_res[name][0] += val * sup['c']; reac_res[name][1] += val * sup['s']
            else:
                reac_res[name][0] -= val * sup['s']; reac_res[name][1] += val * sup['c']
                
        return mem_res, {k: tuple(v) for k, v in reac_res.items()}

# ==============================================================================
# PHẦN 2: GIAO DIỆN WEB
# ==============================================================================
st.set_page_config(page_title="Phân tích Giàn 2D", layout="wide")

if 'truss' not in st.session_state: st.session_state.truss = TrussSolver()
if 'solution' not in st.session_state: st.session_state.solution = None
if 'script_content' not in st.session_state: st.session_state.script_content = ""

truss = st.session_state.truss

def parse_script(text):
    truss.clear_all()
    st.session_state.solution = None
    lines = text.split('\n')
    logs = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts or parts[0].startswith("#"): continue
        cmd = parts[0].upper()
        try:
            if cmd == "NODE": 
                truss.add_node(parts[1].upper(), parts[2], parts[3])
            elif cmd in ["BAR", "MEMBER"]:
                if len(parts) == 2: n1, n2 = parts[1][0], parts[1][1]
                else: n1, n2 = parts[1], parts[2]
                truss.add_member(n1.upper(), n2.upper())
            elif cmd in ["PIN", "ROLLER"]:
                ang = parts[2] if len(parts) > 2 else 0
                truss.add_support(cmd.lower(), parts[1].upper(), ang)
            elif cmd == "LOAD":
                truss.add_load(parts[1].upper(), parts[2], parts[3])
            elif cmd == "SOLVE":
                mem, reac = truss.solve()
                st.session_state.solution = (mem, reac)
                logs.append("✅ Đã giải xong hệ!")
        except Exception as e:
            logs.append(f"❌ Lỗi dòng '{line}': {e}")
            
    return logs

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Điều khiển")
    
    # Mặc định luôn là Script Mode
    st.info("Nhập lệnh bên dưới hoặc bấm 'Tải Ví Dụ' để xem mẫu.")
    
    # --- ĐÂY LÀ ĐOẠN CODE BẠN YÊU CẦU ---
    example_code = """# Ví dụ giàn 
NODE A 0 0
NODE B 4 0 
NODE C 8 0 
NODE D 12 0
NODE E 6 6
BAR AB
BAR BC
BAR CD
BAR AE
BAR BE
BAR CE
BAR DE
PIN A 0
ROLLER D 0
LOAD B 3 270
LOAD C 6 270
SOLVE"""
    # -------------------------------------
    
    if st.button("Tải Ví Dụ Mẫu"):
        st.session_state.script_content = example_code
    
    # Text Area
    script_text = st.text_area("Khung nhập code:", value=st.session_state.script_content, height=400, key="script_box")
    
    if st.button("▶️ CHẠY SCRIPT (SOLVE)", type="primary"):
        logs = parse_script(script_text)
        if logs:
            with st.expander("Nhật ký chạy (Logs)", expanded=True):
                for log in logs: st.write(log)
    
    st.divider()
    if st.button("Xóa trắng"): 
        st.session_state.script_content = ""
        truss.clear_all()
        st.session_state.solution = None
        st.rerun()

# --- MAIN DISPLAY ---
st.header("Mô phỏng kết cấu")

c_left, c_right = st.columns([3, 1])

with c_left:
    if truss.nodes:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ thanh
        for m in truss.members:
            x = [m.node_i.x, m.node_j.x]; y = [m.node_i.y, m.node_j.y]
            col = 'black'; lw = 2
            if st.session_state.solution:
                f = st.session_state.solution[0].get(m.name, 0)
                if f > 0.001: col='blue'; lw=3
                elif f < -0.001: col='red'; lw=3
                
                # Hiện số lực
                ax.text(np.mean(x), np.mean(y), f"{f:.2f}", color=col, fontsize=9, fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                
            ax.plot(x, y, color=col, linewidth=lw, marker='o', mfc='white', mec='black')
        
        # Vẽ nút
        for n in truss.nodes.values(): ax.text(n.x, n.y+0.3, n.name, fontweight='bold', ha='center', fontsize=12)
        
        # Vẽ gối
        for s in truss.supports: 
            ax.plot(truss.nodes[s].x, truss.nodes[s].y-0.2, '^', color='gray', ms=12)
            
        # Vẽ tải (Mũi tên đỏ)
        for l in truss.loads:
            n = truss.nodes[l['node']]
            ax.annotate("", xy=(n.x, n.y), xytext=(n.x, n.y+1.5),
                        arrowprops=dict(facecolor='red', width=2, headwidth=8))
            ax.text(n.x, n.y+1.6, f"{l['P']}kN", ha='center', color='red')

        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
    else:
        st.info("👈 Hãy bấm 'Tải Ví Dụ Mẫu' rồi bấm 'CHẠY SCRIPT' ở menu bên trái.")

with c_right:
    if st.session_state.solution:
        st.success("Kết quả tính toán")
        mem, reac = st.session_state.solution
        
        st.write("**Nội lực (kN):**")
        # Format lại bảng cho đẹp
        data = []
        for k, v in mem.items():
            trang_thai = "Kéo" if v > 0.001 else "Nén" if v < -0.001 else "-"
            data.append({"Thanh": k, "Lực": f"{v:.2f}", "TT": trang_thai})
        st.dataframe(data, hide_index=True)
        
        st.write("**Phản lực (kN):**")
        for k, v in reac.items(): st.write(f"📍 {k}: X={v[0]:.2f}, Y={v[1]:.2f}")
