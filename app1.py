import streamlit as st
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import io  

st.set_page_config(page_title="データ解析トレーニング", layout="wide")
st.title("🌱 農業データ解析トレーニング（GLM ＆ ノンパラ編）")
st.markdown("このアプリは「正しい結果を出す」ことではなく、「**データ構造に合わせて適切な解析手法を選び、結果を正しく解釈できるか**」を鍛えるためのシミュレーターです。")

# ==========================================
# 状態管理（Streamlitの再実行ループ対策）
# ==========================================
if "df" not in st.session_state:
    st.session_state.df = None

# ==========================================
# 埋め込みサンプルデータ（CSV形式の文字列）
# ==========================================
csv_binomial = """品種,調査個体数,腐敗数,積算温度
とうや,141,5,1242.9
とうや,120,8,1350.5
とうや,142,16,1515.8
とうや,130,14,1515.8
とうや,174,42,1684.7
とうや,155,58,1800.2
男爵,150,2,1242.9
男爵,135,3,1350.5
男爵,140,8,1515.8
男爵,125,7,1515.8
男爵,160,25,1684.7
男爵,145,35,1800.2
メークイン,145,10,1242.9
メークイン,130,15,1350.5
メークイン,138,28,1515.8
メークイン,128,25,1515.8
メークイン,165,60,1684.7
メークイン,152,85,1800.2"""

csv_poisson = """トラップ種類,温度,捕獲数
フェロモンA,15,2
フェロモンA,20,5
フェロモンA,25,12
フェロモンA,30,28
フェロモンA,35,45
フェロモンB,15,0
フェロモンB,20,2
フェロモンB,25,4
フェロモンB,30,8
フェロモンB,35,15"""

csv_nonparam = """品種,処理区,発病指数
とうや,無処理,3
とうや,無処理,4
とうや,無処理,4
とうや,無処理,3
とうや,農薬A,1
とうや,農薬A,0
とうや,農薬A,1
とうや,農薬A,1
男爵,無処理,2
男爵,無処理,2
男爵,無処理,3
男爵,無処理,2
男爵,農薬A,0
男爵,農薬A,1
男爵,農薬A,0
男爵,農薬A,0"""

# ==========================================
# STEP 1: データ読み込み（タブで切り替え）
# ==========================================
st.header("Step1：データの読み込み")

tab1, tab2 = st.tabs(["📝 アプリ内蔵のサンプルデータを使う", "📁 手持ちのCSVをアップロードする"])

with tab1:
    st.write("研修用に用意されたデータセットをワンクリックで読み込んで試すことができます。")
    sample_choice = st.selectbox(
        "使用するサンプルデータを選択してください", 
        ["🥔 腐敗率データ（二項分布の練習用）", "🐛 害虫捕獲データ（ポアソン分布の練習用）", "🌿 発病指数データ（ノンパラ検定の練習用）"]
    )
    
    if st.button("サンプルデータを読み込む", type="primary"):
        if "二項分布" in sample_choice:
            st.session_state.df = pd.read_csv(io.StringIO(csv_binomial))
        elif "ポアソン分布" in sample_choice:
            st.session_state.df = pd.read_csv(io.StringIO(csv_poisson))
        else:
            st.session_state.df = pd.read_csv(io.StringIO(csv_nonparam))
        st.success(f"✅ 「{sample_choice}」を読み込みました！下にスクロールしてStep2に進んでください。")

with tab2:
    file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if file:
        col1, col2 = st.columns([2, 1])
        with col1:
            encoding = st.selectbox("文字コード（日本のExcelで作成した場合はcp932を推奨）", ["utf-8", "cp932"])
        with col2:
            st.write("")
            st.write("")
            if st.button("読み込み実行", key="load_btn"):
                try:
                    file.seek(0)
                    st.session_state.df = pd.read_csv(file, encoding=encoding)
                    st.success("✅ アップロードされたデータを読み込みました！")
                except Exception as e:
                    st.error("🚨 読み込みエラー：文字コードが合っていない可能性があります。")
                    st.code(str(e))

# ==========================================
# STEP 2 & 3: 解析設定
# ==========================================
if st.session_state.df is not None:
    df = st.session_state.df.copy()

    st.markdown("---")
    st.header("Step2：変数の役割定義")
    st.write("データプレビュー (先頭5行):")
    st.dataframe(df.head())

    cols = df.columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        col_total = st.selectbox("母数（総調査個体数 n など） ※上限がない、または指数データの場合は「なし」", ["なし"] + cols)
        col_event = st.selectbox("目的変数（発芽数、虫の数、発病指数 など）", cols, index=min(1, len(cols)-1))
    with c2:
        col_exp1 = st.selectbox("説明変数1（品種、処理区 など）", cols, index=min(2, len(cols)-1))
        col_exp2 = st.selectbox("説明変数2（温度、肥料 など）", ["なし"] + cols)

    st.markdown("---")
    st.header("Step3：モデルの選択")
    
    dist = st.radio(
        "このデータが従う分布（手法）は何を仮定しますか？", 
        [
            "正規分布 (OLS: 通常の線形回帰)", 
            "二項分布 (GLM: 上限のある割合データ)", 
            "ポアソン分布 (GLM: 上限のないカウントデータ)",
            "ノンパラメトリック検定 (Kruskal-Wallis検定: 順序データや正規性のないデータ)"
        ]
    )
    interaction = st.radio("説明変数間の交互作用を考慮しますか？", ["なし", "あり（変数1 × 変数2）"])

    if st.button("解析を実行して評価する", key="analyze_btn", type="primary"):

        # ==========================================
        # 教育のための関所（入力値バリデーション）
        # ==========================================
        error_found = False

        if col_total != "なし" and col_total == col_event:
            st.error("🚨 【考え直してみましょう】母数と目的変数に同じ列が選ばれています。")
            error_found = True
            
        if col_event == col_exp1 or col_event == col_exp2:
            st.error("🚨 【考え直してみましょう】目的変数と説明変数に同じ列が選ばれています。原因（説明変数）と結果（目的変数）は別のデータである必要があります。")
            error_found = True

        if col_exp2 != "なし" and col_exp1 == col_exp2:
            st.error("🚨 【考え直してみましょう】説明変数1と説明変数2に同じ列が選ばれています。")
            error_found = True

        if interaction == "あり（変数1 × 変数2）" and col_exp2 == "なし":
            st.warning("⚠️ 説明変数2が「なし」のため、交互作用なしとして計算を続行します。")
            interaction = "なし"

        if error_found:
            st.stop()

        if not pd.api.types.is_numeric_dtype(df[col_event]):
            st.error("🚨 【考え直してみましょう】目的変数に「文字データ（品種など）」が選ばれています。解析対象は数値である必要があります。")
            error_found = True

        if col_total != "なし" and not pd.api.types.is_numeric_dtype(df[col_total]):
            st.error("🚨 【考え直してみましょう】母数に「文字データ（品種など）」が選ばれています。母数は数値（カウント）である必要があります。")
            error_found = True

        if error_found:
            st.stop()

        if "二項分布" in dist:
            if col_total == "なし":
                st.error("🚨 【考え直してみましょう】二項分布には「母数（分母）」が必要です。")
                error_found = True
            elif pd.api.types.is_float_dtype(df[col_total]):
                st.error("🚨 【考え直してみましょう】二項分布の母数に連続値（小数）が選ばれています。母数は整数である必要があります。")
                error_found = True
            elif (df[col_event] > df[col_total]).any():
                st.error("🚨 【考え直してみましょう】目的変数が母数を上回っている行があります。確率は100%（1.0）を超えません。")
                error_found = True

        elif "ポアソン分布" in dist:
            if col_total != "なし":
                st.error("🚨 【考え直してみましょう】ポアソン分布は上限が不明なデータに使います。母数は「なし」にしてください。")
                error_found = True
            elif (df[col_event] < 0).any() or (df[col_event] % 1 != 0).any():
                st.error("🚨 【考え直してみましょう】ポアソン分布の目的変数は0以上の「整数（カウント）」である必要があります。")
                error_found = True

        elif "ノンパラメトリック" in dist:
            if col_total != "なし":
                st.error("🚨 【考え直してみましょう】ノンパラメトリック検定では「母数」の指定は不要です。")
                error_found = True
            elif col_exp2 != "なし":
                st.error("🚨 【考え直してみましょう】Kruskal-Wallis検定は「一元配置（説明変数が1つ）」の検定です。説明変数2は「なし」に設定してください。")
                error_found = True
            elif pd.api.types.is_float_dtype(df[col_exp1]) or (pd.api.types.is_numeric_dtype(df[col_exp1]) and df[col_exp1].nunique() > 6):
                st.error("🚨 【考え直してみましょう】説明変数1に「連続値（温度など）」や「ばらつきの大きい数値（捕獲数など）」が選ばれています。Kruskal-Wallis検定は『グループ間』の比較を行う手法です。回帰分析（GLMなど）を使用してください。")
                error_found = True

        if error_found:
            st.stop() 

        # ==========================================
        # 解析処理と詳細なフィードバック
        # ==========================================
        try:
            st.markdown("---")
            st.header("📊 解析結果とフィードバック")

            if "ノンパラメトリック" in dist:
                groups = [group[col_event].dropna() for name, group in df.groupby(col_exp1)]
                stat, p_val = stats.kruskal(*groups)
                
                st.success("🎉 【正解】データ構造に合わせた適切な手法（ノンパラメトリック検定）が選択されました。")
                st.markdown("#### 📖 なぜこの選択が正しいのか？")
                st.info("発病指数（0=無症状、1=微小、2=拡大...）や食味評価のような「順序データ」は、数値の間隔に連続的な意味がありません。これを無理に正規分布とみなして平均値を比較すると、実態と異なる結論を導く危険があります。ノンパラメトリック検定は、データを「順位」に変換して比較するため、安全に評価できます。")
                st.markdown("#### 🔍 解析結果の読み方")
                st.write(f"- **検定統計量 (H)**: {stat:.4f} （グループ間の順位のズレの大きさ）")
                st.write(f"- **p値**: **{p_val:.4g}**")
                
                if p_val < 0.05:
                    st.success(f"👉 **結論**: p値が0.05未満のため、グループ間に「統計的に有意な差がある」と判断できます。")
                else:
                    st.warning(f"👉 **結論**: p値が0.05以上のため、グループ間に「有意な差があるとは言えない」と判断されます。")

            else:
                if col_exp2 == "なし":
                    exp_formula = f"{col_exp1}"
                else:
                    exp_formula = f"{col_exp1} + {col_exp2}"
                    if interaction != "なし":
                        exp_formula += f" + {col_exp1}:{col_exp2}"

                if "二項分布" in dist:
                    df['proportion'] = df[col_event] / df[col_total]
                    formula = f"proportion ~ {exp_formula}"
                    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), var_weights=df[col_total]).fit()
                    
                    st.success("🎉 【正解】「上限のある割合データ」に対して二項分布（GLM）を適切に選択できました。")
                    st.markdown("#### 📖 なぜこの選択が正しいのか？")
                    st.info("発芽率や腐敗率などのデータは「必ず0%〜100%の間に収まる」という制約があり、確率が50%に近いほど分散（ばらつき）が大きくなる性質があります。GLM（ロジットリンク関数）を使うことで、予測値が0未満や100%以上になることを防ぎ、分散の偏りも正しく処理できます。")
                    st.write(model.summary())
                    st.markdown("#### 🔍 解析結果（サマリー表）の読み方")
                    st.markdown("""
                    農業論文を書く上で、表の真ん中にある **`coef`** と **`P>|z|`** の列が最も重要です。
                    * **`coef` (偏回帰係数)**: その変数が結果に与える「影響の向きと大きさ」です。プラスなら事象（腐敗など）を増やし、マイナスなら減らす働きがあります。
                    * **`P>|z|` (p値)**: その変数の影響が「誤差（偶然）ではない」と言える確率の指標です。一般的に **p < 0.05** であれば「有意な影響がある（意味のある差である）」とみなします。
                    * **`[0.025  0.975]` (95%信頼区間)**: 真の係数が95%の確率で収まる範囲です。この範囲が「0」を跨いでいなければ、有意差ありと同義です。
                    """)

                elif "ポアソン分布" in dist:
                    formula = f"{col_event} ~ {exp_formula}"
                    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
                    
                    st.success("🎉 【正解】「上限のないカウントデータ」に対してポアソン分布（GLM）を適切に選択できました。")
                    st.markdown("#### 📖 なぜこの選択が正しいのか？")
                    st.info("害虫の発見数や雑草の発生数など、「発生確率は低いが試行回数が多い（上限が不明な）事象」はポアソン分布に従います。データは「必ず0以上の整数」になり、平均値が大きくなるほど分散も大きくなる性質があります。GLM（対数リンク関数）を使うことで、予測値がマイナスになる理論的破綻を防げます。")
                    st.write(model.summary())
                    st.markdown("#### 🔍 解析結果（サマリー表）の読み方")
                    st.markdown("""
                    * **`coef` (偏回帰係数)**: GLM(ポアソン)では対数変換されているため、そのままの数値ではなく「自然指数(e)の累乗」として影響を解釈します。
                    * **`P>|z|` (p値)**: その変数がカウント数に「有意な影響を与えているか」を示します。**p < 0.05** であれば統計的に意味のある要因です。
                    """)

                # 修正部分：正規分布（OLS）の評価ロジック
                else:
                    # ① 割合データ（母数あり）に正規分布を選んだ場合（不正解トラップ）
                    if col_total != "なし":
                        formula_ols = f"I({col_event}/{col_total}) ~ {exp_formula}"
                        model = smf.ols(formula=formula_ols, data=df).fit()
                        
                        st.error("❌ 【不正解】変数の割り当ては合っていますが、モデル（分布）の選択が不適切です。")
                        st.markdown("#### 🤔 なぜこのモデルはダメなのか？（査読で指摘されるポイント）")
                        st.warning("割合データに対して「通常の線形回帰（正規分布）」を無理やり当てはめると、直線で予測を立ててしまうため「予測値が0%未満になる」「100%を超える」といった理論的破綻が起きます。また等分散性の前提も崩れるため、二項分布（GLM）を選択してください。")
                        st.write(model.summary())
                        
                    # ② カウントデータ（値が小さい整数）に正規分布を選んだ場合（不正解トラップ）
                    elif (df[col_event] % 1 == 0).all() and (df[col_event] >= 0).all() and df[col_event].max() < 100:
                        formula_ols = f"{col_event} ~ {exp_formula}"
                        model = smf.ols(formula=formula_ols, data=df).fit()
                        
                        st.error("❌ 【不正解】変数の割り当ては合っていますが、モデル（分布）の選択が不適切です。")
                        st.markdown("#### 🤔 なぜこのモデルはダメなのか？（査読で指摘されるポイント）")
                        st.warning("カウントデータ（特に平均が小さいもの）に「通常の線形回帰（正規分布）」を当てはめると、予測値がマイナスになる理論的破綻が起きます。また、カウントデータは平均が大きくなるほど分散も大きくなる性質があるため、ポアソン分布（GLM）を選択してください。")
                        st.write(model.summary())
                        
                    # ③ 連続データ（収量など）に正規分布を選んだ場合（正解！）
                    else:
                        formula_ols = f"{col_event} ~ {exp_formula}"
                        model = smf.ols(formula=formula_ols, data=df).fit()
                        
                        st.success("🎉 【正解】連続データに対して「通常の線形回帰（正規分布）」を適切に選択できました！")
                        st.markdown("#### 📖 なぜこの選択が正しいのか？")
                        st.info("収量や重量、長さなどの「連続変数」は、正規分布を仮定するモデル（通常の線形回帰や分散分析など）で解析するのが基本であり、最も適したアプローチです。")
                        st.write(model.summary())
                        st.markdown("#### 🔍 解析結果（サマリー表）の読み方")
                        st.markdown("""
                        農業論文を書く上で、表の真ん中にある **`coef`** と **`P>|t|`** の列が最も重要です。
                        * **`coef` (偏回帰係数)**: その変数が目的変数（収量など）に与える「影響の向きと大きさ」です。
                        * **`P>|t|` (p値)**: その変数の影響が「誤差（偶然）ではない」と言える確率の指標です。一般的に **p < 0.05** であれば「有意な影響がある」とみなします。
                        * **`[0.025  0.975]` (95%信頼区間)**: 真の係数が95%の確率で収まる範囲です。この範囲が「0」を跨いでいなければ、有意差ありと同義です。
                        """)

        except Exception as e:
            st.error("解析中に予期せぬエラーが発生しました。データの構造を確認してください。")
            st.code(str(e))
