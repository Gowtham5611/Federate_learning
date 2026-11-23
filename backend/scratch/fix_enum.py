from pathlib import Path

path = Path("backend/domain/enums/model_architecture.py")
content = path.read_text(encoding="utf-8")

old_enum = """    GRU = "gru"
    LSTM = "lstm"
    BILSTM = "bilstm"
    TCN = "tcn"
    TRANSFORMER = "transformer"
    RESIDUAL_MLP = "residual_mlp" """

new_enum = """    GRU = "gru"
    LSTM = "lstm"
    BILSTM = "bilstm" """

lines = content.splitlines()
new_lines = []
for line in lines:
    if any(k in line for k in ["TCN =", "TRANSFORMER =", "RESIDUAL_MLP =", "ModelArchitecture.TCN:", "ModelArchitecture.TRANSFORMER:", "ModelArchitecture.RESIDUAL_MLP:"]):
        continue
    new_lines.append(line)

path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
print("Enum successfully cleaned to GRU, LSTM, BILSTM")

