
import io
import json
from typing import Dict, List
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

PDF_OK = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
except ImportError:
    PDF_OK = False

APP_TITLE = "Seleção com Ações Afirmativas — V5.1 (Aprovação + Espera, com Remanejamento entre AAs)"
INTRO = '''
**O que esta versão faz**
- Calcula as vagas por Ações Afirmativas com **teto exato** (ex.: 50%), usando distribuição proporcional com **mínimo de 1** por grupo com elegíveis.
- **Remanejamento entre AAs**: se um grupo não tiver elegíveis suficientes, suas vagas são redistribuídas **entre os demais grupos de AAs que tenham elegíveis disponíveis**, antes de devolver à **Ampla**.
- Gera **apenas a 1ª chamada** (sem controle de matrícula).
- Exporta:
  1) **Validação do upload** (todos os candidatos e as listas em que concorrem);
  2) **Quatro listas** (AMPLA, PPI, Q, PCD): **Aprovados** e **Lista de Espera** (quem não entrou em nenhuma lista).
'''

def normalize_bool_series(s):
    if s is None:
        return pd.Series(dtype=bool)
    s = s.astype(str).str.strip().str.lower()
    return s.isin(["sim", "true", "1", "y", "yes"])

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "nome": "nome", "name": "nome", "candidate": "nome",
        "rank": "rank", "classificacao": "rank", "classificação": "rank", "posicao": "rank",
        "is_ppi": "is_ppi", "ppi": "is_ppi",
        "is_q": "is_q", "quilombola": "is_q", "q": "is_q",
        "is_pcd": "is_pcd", "pcd": "is_pcd",
    }
    lower_cols = {c.lower(): c for c in df.columns}
    mapped = {}
    for k, v in rename_map.items():
        if k in lower_cols:
            mapped[lower_cols[k]] = v
    df = df.rename(columns=mapped)

    required = ["nome", "rank"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")
    for flag in ["is_ppi", "is_q", "is_pcd"]:
        if flag not in df.columns:
            df[flag] = False
    else:
        for flag in ["is_ppi", "is_q", "is_pcd"]:
            if df[flag].dtype == object:
                df[flag] = normalize_bool_series(df[flag])
            df[flag] = df[flag].fillna(False).astype(bool)

    try:
        df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(int)
    except Exception:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
        if df["rank"].isna().any():
            raise ValueError("A coluna 'rank' contém valores não numéricos.")
        df["rank"] = df["rank"].astype(int)

    df["_orig_order"] = np.arange(len(df))

    return df[["nome", "rank", "is_ppi", "is_q", "is_pcd", "_orig_order"]]

def apportion_with_min_and_redistribution(total: int, perc_aa_pct: float, weights_pct: Dict[str, float],
                                          elig_counts: Dict[str, int]) -> Dict[str, int]:
    groups = ["PPI", "Q", "PCD"]
    aa_target = int(round(total * (perc_aa_pct / 100.0)))

    raw_sum = sum(weights_pct.values())
    if raw_sum <= 0:
        raise ValueError("As porcentagens internas devem somar > 0.")
    w = {k: (v / raw_sum) for k, v in weights_pct.items()}

    mins = {g: (1 if elig_counts.get(g, 0) > 0 else 0) for g in groups}
    sum_min = sum(mins.values())
    if sum_min > aa_target:
        order = sorted(groups, key=lambda g: (w[g], g))
        to_drop = sum_min - aa_target
        for g in order:
            if to_drop <= 0: break
            if mins[g] > 0:
                mins[g] -= 1; to_drop -= 1

    quotas = {g: aa_target * w[g] for g in groups}
    alloc = {g: mins[g] for g in groups}
    remainder = aa_target - sum(alloc.values())
    fracs = {g: quotas[g] - alloc[g] for g in groups}
    while remainder > 0:
        g = max(groups, key=lambda x: (fracs[x] - int(fracs[x]), quotas[x], x))
        alloc[g] += 1
        fracs[g] -= 1
        remainder -= 1

    overflow = 0
    caps = {g: int(elig_counts.get(g, 0)) for g in groups}
    for g in groups:
        cap = caps[g]
        if alloc[g] > cap:
            overflow += (alloc[g] - cap)
            alloc[g] = cap

    def available_capacity(g): return max(0, caps[g] - alloc[g])
    while overflow > 0 and any(available_capacity(g) > 0 for g in groups):
        g = max(
            [x for x in groups if available_capacity(x) > 0],
            key=lambda x: (quotas[x] - alloc[x], quotas[x], x)
        )
        alloc[g] += 1
        overflow -= 1

    aa_final = sum(alloc.values())
    ac = total - aa_final
    return {"PPI": int(alloc["PPI"]), "Q": int(alloc["Q"]), "PCD": int(alloc["PCD"]), "AC": int(ac), "AA_alvo": aa_target, "weights_norm": w}

def build_base(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base = base.sort_values(by=["rank", "_orig_order"], ascending=[True, True]).reset_index(drop=True)
    return base

def allocate_first_call(df: pd.DataFrame, seats: Dict[str, int]) -> Dict[str, pd.DataFrame]:
    base = build_base(df)
    chosen_idx = set()
    picks = {k: pd.DataFrame(columns=df.columns) for k in ["AMPLA", "PPI", "Q", "PCD"]}

    def take(category: str, n: int, mask: pd.Series) -> pd.DataFrame:
        if n <= 0: return pd.DataFrame(columns=base.columns)
        avail = base[~base.index.isin(chosen_idx)].copy()
        subset = avail[mask.loc[avail.index]].head(n).copy()
        chosen_idx.update(subset.index.tolist())
        subset["categoria"] = category
        subset["ordem_na_lista"] = np.arange(1, len(subset) + 1)
        return subset

    m_all = pd.Series(True, index=base.index)
    m_ppi = base["is_ppi"]
    m_q = base["is_q"]
    m_pcd = base["is_pcd"]

    n_ac = max(0, seats.get("AC", 0))
    n_ppi = max(0, seats.get("PPI", 0))
    n_q = max(0, seats.get("Q", 0))
    n_pcd = max(0, seats.get("PCD", 0))

    picks["AMPLA"] = take("AMPLA", n_ac, m_all)
    picks["PPI"]   = take("PPI", n_ppi, m_ppi)
    picks["Q"]     = take("Q", n_q, m_q)
    picks["PCD"]   = take("PCD", n_pcd, m_pcd)

    return picks

def build_waitlists(df: pd.DataFrame, picks: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    base = build_base(df)
    approved_names = set(pd.concat([picks[k] for k in picks], ignore_index=True)["nome"].tolist())
    remaining = base[~base["nome"].isin(approved_names)].copy()

    wl = {}
    wl["AMPLA"] = remaining[["nome","rank","_orig_order"]].copy()
    wl["AMPLA"]["ordem_espera"] = np.arange(1, len(wl["AMPLA"]) + 1)

    for cat, mask_col in [("PPI","is_ppi"), ("Q","is_q"), ("PCD","is_pcd")]:
        subset = remaining[remaining[mask_col]].copy()
        subset = subset[["nome","rank","_orig_order"]].copy()
        subset["ordem_espera"] = np.arange(1, len(subset) + 1)
        wl[cat] = subset

    return wl

def df_to_pdf_bytes(title: str, subtitle: str, sections: List[Dict]) -> bytes:
    if not PDF_OK:
        raise RuntimeError("Biblioteca 'reportlab' não instalada.")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=30, rightMargin=30, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Paragraph(subtitle, styles["Normal"]))
    story.append(Spacer(1, 12))

    for sec in sections:
        story.append(Paragraph(f"<b>{sec['title']}</b>", styles["Heading2"]))
        df = sec["df"]
        if df.empty:
            story.append(Paragraph("Sem registros.", styles["Normal"]))
        else:
            cols = list(df.columns)
            data = [cols] + df.values.tolist()
            table = Table(data, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("ALIGN", (0,0), (-1,-1), "LEFT"),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightcyan]),
            ]))
            story.append(table)
        story.append(Spacer(1, 10))
        if sec.get("page_break"):
            story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

def save_project_json(df, params, seats, picks, waitlists) -> bytes:
    state = {
        "df": df.to_dict(orient="records") if df is not None else None,
        "params": params,
        "seats": seats,
        "picks": {k: v.to_dict(orient="records") for k, v in picks.items()} if picks else {},
        "waitlists": {k: v.to_dict(orient="records") for k, v in waitlists.items()} if waitlists else {},
        "saved_at": datetime.now().isoformat()
    }
    return json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")

def load_project_json(file_bytes: bytes):
    obj = json.loads(file_bytes.decode("utf-8"))
    df = pd.DataFrame(obj.get("df", [])) if obj.get("df") else None
    seats = obj.get("seats", {})
    params = obj.get("params", {})
    picks = {k: pd.DataFrame(v) for k, v in obj.get("picks", {}).items()}
    waitlists = {k: pd.DataFrame(v) for k, v in obj.get("waitlists", {}).items()}
    return df, params, seats, picks, waitlists

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("1ª chamada (aprovação + espera) com remanejamento entre AAs; exportações em PDF/XLSX.")
st.markdown(INTRO)

if "df" not in st.session_state:
    st.session_state.df = None
if "params" not in st.session_state:
    st.session_state.params = {}
if "seats" not in st.session_state:
    st.session_state.seats = {}
if "picks" not in st.session_state:
    st.session_state.picks = {}
if "waitlists" not in st.session_state:
    st.session_state.waitlists = {}

with st.sidebar:
    st.subheader("Projeto")
    if st.session_state.df is not None:
        st.download_button("Salvar projeto (.json)",
                           data=save_project_json(st.session_state.df, st.session_state.params, st.session_state.seats, st.session_state.picks, st.session_state.waitlists),
                           file_name="projeto_afirmativas_v51.json", mime="application/json")
    up_proj = st.file_uploader("Carregar projeto (.json)", type=["json"], key="up_proj_v51")
    if up_proj is not None:
        df, params, seats, picks, waitlists = load_project_json(up_proj.read())
        st.session_state.df = df
        st.session_state.params = params
        st.session_state.seats = seats
        st.session_state.picks = picks
        st.session_state.waitlists = waitlists
        st.success("Projeto carregado.")

with st.expander("1) Importar dados (.csv ou .xlsx)", expanded=True):
    up = st.file_uploader("Selecione o arquivo", type=["csv","xlsx"], key="up_data_v51")
    if up is not None:
        try:
            if up.name.endswith(".csv"):
                df_raw = pd.read_csv(up)
            else:
                df_raw = pd.read_excel(up)
            df = ensure_columns(df_raw)
            st.session_state.df = df
            st.success(f"{len(df)} linhas importadas.")
            st.dataframe(df, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao importar: {e}")

if st.session_state.df is None:
    st.stop()

df = st.session_state.df

with st.expander("2) Parâmetros", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    total = c1.number_input("Total de vagas", min_value=1, value=50, step=1)
    perc_aa_pct = c2.number_input("% de Ações Afirmativas (0–100)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    w_ppi_pct = c3.number_input("% PPI dentro das AAs (0–100)", min_value=0.0, max_value=100.0, value=59.0, step=1.0)
    w_q_pct   = c4.number_input("% Q dentro das AAs (0–100)",   min_value=0.0, max_value=100.0, value=1.0, step=1.0)
    w_pcd_pct = c5.number_input("% PCD dentro das AAs (0–100)", min_value=0.0, max_value=100.0, value=7.0, step=1.0)

    auto_norm = st.checkbox("Ajustar para somar 100% (normalizar)")
    weights_pct = {"PPI": w_ppi_pct, "Q": w_q_pct, "PCD": w_pcd_pct}
    if auto_norm:
        s = sum(weights_pct.values())
        if s > 0:
            weights_pct = {k: (v / s) * 100.0 for k, v in weights_pct.items()}
            st.info(f"Normalizado: PPI={weights_pct['PPI']:.2f}%, Q={weights_pct['Q']:.2f}%, PCD={weights_pct['PCD']:.2f}%")

    elig_counts = {"PPI": int(df["is_ppi"].sum()), "Q": int(df["is_q"].sum()), "PCD": int(df["is_pcd"].sum())}
    st.write(f"Elegíveis: PPI={elig_counts['PPI']}, Q={elig_counts['Q']}, PCD={elig_counts['PCD']}")

    if st.button("Calcular 1ª chamada"):
        seats = apportion_with_min_and_redistribution(total, perc_aa_pct, weights_pct, elig_counts)
        st.session_state.seats = seats
        st.session_state.params = {"total": total, "perc_aa_pct": perc_aa_pct, "weights_pct": weights_pct}
        picks = allocate_first_call(df, seats)
        waitlists = build_waitlists(df, picks)
        st.session_state.picks = picks
        st.session_state.waitlists = waitlists
        st.success(f"Vagas → AC: {seats['AC']} | PPI: {seats['PPI']} | Q: {seats['Q']} | PCD: {seats['PCD']} (AA alvo: {seats['AA_alvo']})")

if st.session_state.picks:
    seats = st.session_state.seats
    picks = st.session_state.picks
    wait = st.session_state.waitlists

    st.subheader("3) Visualização — Aprovados e Lista de Espera")
    tabs = st.tabs(["AMPLA", "PPI", "Q", "PCD", "Validação do Upload"])

    cats = ["AMPLA","PPI","Q","PCD"]

    for tab, cat in zip(tabs[:4], cats):
        with tab:
            st.markdown(f"**Vagas {cat}: {seats['AC' if cat=='AMPLA' else cat]}**")
            ap = picks[cat][["ordem_na_lista","nome","rank"]].rename(columns={"ordem_na_lista":"Ordem","rank":"Classificação"})
            st.markdown("**Aprovados**")
            st.dataframe(ap, hide_index=True, use_container_width=True)
            wl = wait[cat][["ordem_espera","nome","rank"]].rename(columns={"ordem_espera":"Ordem Espera","rank":"Classificação"})
            st.markdown("**Lista de Espera**")
            st.dataframe(wl, hide_index=True, use_container_width=True)

    with tabs[4]:
        val = df.copy()
        val["Concorre_AMPLA"] = True
        val["Concorre_PPI"] = val["is_ppi"]
        val["Concorre_Q"]   = val["is_q"]
        val["Concorre_PCD"] = val["is_pcd"]
        val_view = val[["nome","rank","Concorre_AMPLA","Concorre_PPI","Concorre_Q","Concorre_PCD"]]
        st.dataframe(val_view.sort_values(["rank","nome"]), hide_index=True, use_container_width=True)

    st.subheader("4) Exportações")

    if st.button("Exportar — Validação do Upload (PDF)"):
        if not PDF_OK:
            st.warning("PDF indisponível: instale 'reportlab'.")
        else:
            vv = val_view.rename(columns={"rank":"Classificacao"})
            sections = [{
                "title": "Validação do Upload — Candidatos e Listas em que Concorrem",
                "df": vv,
            }]
            pdf_bytes = df_to_pdf_bytes("Validação do Upload", f"Gerado em {datetime.now():%Y-%m-%d %H:%M}", sections)
            st.download_button("Baixar PDF (Validação)", data=pdf_bytes, file_name="validacao_upload.pdf", mime="application/pdf")

    if st.button("Exportar — Validação do Upload (XLSX)"):
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            val_view.to_excel(writer, index=False, sheet_name="Validacao")
        st.download_button("Baixar XLSX (Validação)", data=output.getvalue(), file_name="validacao_upload.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.button("Exportar — 4 listas (PDF)"):
        if not PDF_OK:
            st.warning("PDF indisponível: instale 'reportlab'.")
        else:
            sections = []
            for cat in cats:
                ap = picks[cat][["ordem_na_lista","nome","rank"]].rename(columns={"ordem_na_lista":"Ordem","rank":"Classificacao"})
                wl = wait[cat][["ordem_espera","nome","rank"]].rename(columns={"ordem_espera":"Ordem_Espera","rank":"Classificacao"})
                sections.append({"title": f"{cat} — Aprovados", "df": ap})
                sections.append({"title": f"{cat} — Lista de Espera", "df": wl, "page_break": True})
            pdf_bytes = df_to_pdf_bytes("Listas — 1ª Chamada", f"Gerado em {datetime.now():%Y-%m-%d %H:%M}", sections)
            st.download_button("Baixar PDF (4 listas)", data=pdf_bytes, file_name="listas_primeira_chamada.pdf", mime="application/pdf")

    if st.button("Exportar — 4 listas (XLSX)"):
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for cat in cats:
                ap = picks[cat][["ordem_na_lista","nome","rank"]].rename(columns={"ordem_na_lista":"Ordem","rank":"Classificacao"})
                wl = wait[cat][["ordem_espera","nome","rank"]].rename(columns={"ordem_espera":"Ordem_Espera","rank":"Classificacao"})
                ap.to_excel(writer, index=False, sheet_name=f"{cat}_Aprovados")
                wl.to_excel(writer, index=False, sheet_name=f"{cat}_Espera")
        st.download_button("Baixar XLSX (4 listas)", data=output.getvalue(), file_name="listas_primeira_chamada.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
