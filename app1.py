import streamlit as st
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import io
import keyword  

st.set_page_config(page_title="データ解析トレーニング", layout="wide")
st.title("🌱 農業データ解析トレーニング（GLM ＆ ノンパラ編）")
st.markdown("このアプリは「正しい結果を出す」ことではなく、「**データ構造に合わせて適切な解析手法を選び、結果を正しく解釈できるか**」を鍛えるためのシミュレーターです。")

# ==========================================
# 状態管理
# ==========================================
if "df" not in st.session_state:
    st.session_state.df = None

# ==========================================
# 埋め込みサンプルデータ群
# ==========================================
csv_binomial1 = """品種,調査個体数,腐敗数,積算温度
とうや,141,5,1242.9
とうや,120,8,1350.5
とうや,142,16,1515.8
とうや,174,42,1684.7
男爵,150,2,1242.9
男爵,135,3,1350.5
男爵,140,8,1515.8
男爵,160,25,1684.7
メークイン,145,10,1242.9
メークイン,130,15,1350.5
メークイン,138,28,1515.8
メークイン,165,60,1684.7"""

csv_binomial2 = """処理区,播種数,発芽数,気温
無処理,100,65,15
無処理,100,72,20
無処理,100,85,25
温湯消毒,100,80,15
温湯消毒,100,88,20
温湯消毒,100,95,25
薬剤処理,100,85,15
薬剤処理,100,92,20
薬剤処理,100,98,25"""

csv_binomial3 = """薬剤,濃度,総虫数,死亡数
農薬A,100,50,15
農薬A,200,48,25
農薬A,500,52,45
農薬B,100,55,10
農薬B,200,51,18
農薬B,500,49,35"""

csv_poisson1 = """トラップ種類,温度,捕獲数
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

csv_poisson2 = """処理区,土壌湿度,雑草数
無処理,30,15
無処理,40,22
無処理,50,35
除草剤A,30,2
除草剤A,40,4
除草剤A,50,8
除草剤B,30,5
除草剤B,40,7
除草剤B,50,12"""

csv_poisson3 = """品種,葉位,アブラムシ数
品種A,上部,2
品種A,中部,5
品種A,下部,12
品種B,上部,0
品種B,中部,2
品種B,下部,6
品種C,上部,5
品種C,中部,15
品種C,下部,30"""

csv_normal1 = """品種,肥料,収量
A,標準,4.5
A,標準,4.8
A,標準,5.1
A,多肥,6.2
A,多肥,6.5
A,多肥,6.0
B,標準,3.8
B,標準,4.1
B,標準,3.9
B,多肥,5.0
B,多肥,5.2
B,多肥,4.8"""

csv_normal2 = """温度,日照時間,草丈
15,8,12.5
15,10,14.2
15,12,16.0
20,8,18.1
20,10,21.5
20,12,24.3
25,8,22.0
25,10,25.8
25,12,28.5"""

csv_normal3 = """品種,土壌水分,重量
紅あずま,20,250.5
紅あずま,30,310.2
紅あずま,40,345.8
鳴門金時,20,220.3
鳴門金時,30,280.1
鳴門金時,40,305.6
安納芋,20,180.4
安納芋,30,210.5
安納芋,40,240.9"""

csv_nonparam1 = """品種,処理区,発病指数
とうや,無処理,3
とうや,無処理,4
とうや,無処理,3
とうや,農薬A,0
とうや,農薬A,1
とうや,農薬A,1
男爵,無処理,2
男爵,無処理,3
男爵,無処理,2
男爵,農薬A,0
男爵,農薬A,1
男爵,農薬A,0"""

csv_nonparam2 = """品種,評価者ID,食味スコア
あまおう,1,5
あまおう,2,4
あまおう,3,5
あまおう,4,4
とちおとめ,1,3
とちおとめ,2,4
とちおとめ,3,3
とちおとめ,4,3
紅ほっぺ,1,4
紅ほっぺ,2,4
紅ほっぺ,3,3
紅ほっぺ,4,4"""

csv_nonparam3 = """保存温度,サンプル,褐変度
5度,1,0
5度,2,1
5度,3,0
5度,4,0
15度,1,2
15度,2,2
15度,3,3
15度,4,2
25度,1,4
25度,2,4
25度,3,4
25度,4,3"""

csv_nonparam4 = """肥料,反復,根張りスコア
無施肥,1,1
無施肥,2,1
無施肥,3,2
無施肥,4,1
化成肥料,1,3
化成肥料,2,2
化成肥料,3,3
化成肥料,4,3
有機肥料,1,2
有機肥料,2,3
有機肥料,3,2
有機肥料,4,2"""

# ==========================================
# 答えが推測できないように、名前からヒントを消して順序もバラバラに配置
# ==========================================
sample_choices = {
    "🍅 トマトの収量データ": csv_normal1,
    "🐛 害虫捕獲データ": csv_poisson1,
    "🥔 ジャガイモの腐敗データ": csv_binomial1,
    "🍓 イチゴの食味スコアデータ": csv_nonparam2,
    "🌿 雑草の発生本数データ": csv_poisson2,
    "🥬 小松菜の草丈データ": csv_normal2,
    "🌱 種子の発芽データ": csv_binomial2,
    "🦠 発病指数データ": csv_nonparam1,
    "🍠 サツマイモの重量データ": csv_normal3,
    "🍎 果実の褐変度データ": csv_nonparam3,
    "🐞 アブラムシの寄生数データ": csv_poisson3,
    "🦟 害虫の死亡データ": csv_binomial3,
    "🌱 根張りスコアデータ": csv_nonparam4,
}


# ==========================================
# STEP 1: データ読み込み
# ==========================================
st.header("Step1：データの読み込み")
tab1, tab2 = st.tabs(["📝 アプリ内蔵のサンプルデータを使う", "📁 手持ちのCSVをアップロードする"])

with tab1:
    st.write("トレーニング用のデータセットを選択してください。（プレビューを見て適切な手法を推測しましょう！）")
    sample_choice_key = st.selectbox("使用するサンプルデータを選択してください", list(sample_choices.keys()))
    
    if st.button("サンプルデータを読み込む", type="primary"):
        st.session_state.df = pd.read_csv(io.StringIO(sample_choices[sample_choice_key]))
        st.success(f"✅ 「{sample_choice_key}」を読み込みました！下にスクロールしてStep2に進んでください。")

with tab2:
    file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if file:
        col1, col2 = st.columns([2, 1])
        with col1:
            encoding = st.selectbox("文字コード（日本のExcelで作成した場合はcp932を推奨）", ["utf-8", "cp932"])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
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
    st.write("💡 **データプレビューを確認し、このデータが持つ性質（連続値か、整数か、割合の元になるデータか等）を見極めてください。**")
    st.dataframe(df.head())

    cols = df.columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        col_total = st.selectbox("母数（総調査個体数 n など） ※上限がない、または指数データの場合は「なし」",["なし"] + cols)
        col_event = st.selectbox("目的変数（発芽数、虫の数、発病指数 など）", cols, index=min(1, len(cols)-1))
    with c2:
        col_exp1 = st.selectbox("説明変数1（品種、処理区 など）", cols, index=min(2, len(cols)-1))
        col_exp2 = st.selectbox("説明変数2（温度、肥料 など）", ["なし"] + cols)

    st.markdown("---")
    st.header("Step3：モデルの選択")
    
    dist = st.radio(
        "このデータが従う分布（手法）は何を仮定しますか？",[
            "正規分布 (OLS: 通常の線形回帰)", 
            "二項分布 (GLM: 上限のある割合データ)", 
            "ポアソン分布 (GLM: 上限のないカウントデータ)",
            "ノンパラメトリック検定 (Kruskal-Wallis検定: 順序データや正規性のないデータ)"
        ]
    )
    interaction = st.radio("説明変数間の交互作用を考慮しますか？",["なし", "あり（変数1 × 変数2）"])

    if st.button("解析を実行して評価する", key="analyze_btn", type="primary"):

        # ==========================================
        # 教育のための関所（バリデーション）
        # ==========================================
        error_found = False

        used_cols =[col_total, col_event, col_exp1, col_exp2]
        for col in used_cols:
            if col != "なし" and keyword.iskeyword(str(col).lower()):
                st.error(f"🚨 【変数名のエラー】列名に「 `{col}` 」という単語が使われています。")
                st.markdown(f"**💡 なぜ解析できないの？（Python特有の罠）**\n農業データでは「**yield**（収量）」などを列名にしがちですが、Pythonにおいて `yield` や `return` は特別な命令語（予約語）です。システムが区別できずエラーを起こします。\n**✅ 解決策**：CSVの列名を `yield_kg` や日本語の `収量` 等に変更してください。")
                error_found = True
        
        if col_total != "なし" and col_total == col_event:
            st.error("🚨 【考え直してみましょう】母数と目的変数に同じ列が選ばれています。")
            error_found = True
            
        if col_event == col_exp1 or col_event == col_exp2:
            st.error("🚨 【考え直してみましょう】目的変数と説明変数に同じ列が選ばれています。原因と結果は別のデータである必要があります。")
            error_found = True

        if col_exp2 != "なし" and col_exp1 == col_exp2:
            st.error("🚨 【考え直してみましょう】説明変数1と説明変数2に同じ列が選ばれています。")
            error_found = True

        if interaction == "あり（変数1 × 変数2）" and col_exp2 == "なし":
            st.warning("⚠️ 説明変数2が「なし」のため、交互作用なしとして計算を続行します。")
            interaction = "なし"

        if not pd.api.types.is_numeric_dtype(df[col_event]):
            st.error("🚨 【考え直してみましょう】目的変数が「文字データ」です。解析対象は数値である必要があります。")
            error_found = True

        if col_total != "なし" and not pd.api.types.is_numeric_dtype(df[col_total]):
            st.error("🚨 【考え直してみましょう】母数が「文字データ」です。数値である必要があります。")
            error_found = True

        if error_found:
            st.stop()

        if "二項分布" in dist:
            if col_total == "なし":
                st.error("🚨 【考え直してみましょう】二項分布には「母数（分母）」が必要です。")
                error_found = True
            elif not (df[col_total].dropna() % 1 == 0).all():
                st.error("🚨 【考え直してみましょう】二項分布の母数に連続値（小数）が含まれています。整数である必要があります。")
                error_found = True
            elif (df[col_event] > df[col_total]).any():
                st.error("🚨 【考え直してみましょう】目的変数が母数を上回っている行があります。確率は100%を超えません。")
                error_found = True

        elif "ポアソン分布" in dist:
            if col_total != "なし":
                st.error("🚨 【考え直してみましょう】ポアソン分布は上限が不明なデータに使います。母数は「なし」にしてください。")
                error_found = True
            elif (df[col_event].dropna() < 0).any() or not (df[col_event].dropna() % 1 == 0).all():
                st.error("🚨 【考え直してみましょう】ポアソン分布の目的変数は0以上の「整数（カウント）」である必要があります。")
                error_found = True

        elif "ノンパラメトリック" in dist:
            if col_total != "なし":
                st.error("🚨 【考え直してみましょう】ノンパラメトリック検定では「母数」の指定は不要です。")
                error_found = True
            elif col_exp2 != "なし":
                st.error("🚨 【考え直してみましょう】ここでは「一元配置（説明変数が1つ）」の検定を想定しています。説明変数2は「なし」に設定してください。")
                error_found = True
            elif pd.api.types.is_float_dtype(df[col_exp1]) or (pd.api.types.is_numeric_dtype(df[col_exp1]) and df[col_exp1].nunique() > 6):
                st.error("🚨 【考え直してみましょう】説明変数1に「連続値」が選ばれています。Kruskal-Wallis検定は『グループ間』の比較を行う手法です。回帰分析を使用してください。")
                error_found = True

        if error_found:
            st.stop() 

        # ==========================================
        # 解析処理・フォーミュラ生成
        # ==========================================
        try:
            st.markdown("---")
            st.header("📊 解析結果とフィードバック")

            def Q(col_name):
                return f"Q('{col_name}')"
            
            if col_exp2 == "なし":
                exp_formula = Q(col_exp1)
            else:
                if interaction == "なし":
                    exp_formula = f"{Q(col_exp1)} + {Q(col_exp2)}"
                else:
                    exp_formula = f"{Q(col_exp1)} * {Q(col_exp2)}"

            # --- 解析分岐 ---
            if "ノンパラメトリック" in dist:
                groups = [group[col_event].dropna() for name, group in df.groupby(col_exp1)]
                stat, p_val = stats.kruskal(*groups)
                
                st.success("🎉 【正解】データ構造に合わせた適切な手法（ノンパラメトリック検定）が選択されました。")
                st.markdown("#### 📖 なぜこの選択が正しいのか？")
                st.info("発病指数や食味評価のような「順序データ」は、数値の間隔に連続的な意味がありません。無理に正規分布とみなして平均値を比較すると実態と異なる結論を導く危険があります。データを「順位」に変換して比較するため安全です。")
                st.markdown("#### 🔍 解析結果の読み方")
                st.write(f"- **検定統計量 (H)**: {stat:.4f} （グループ間の順位のズレの大きさ）")
                st.write(f"- **p値**: **{p_val:.4g}**")
                
                if p_val < 0.05:
                    st.success("👉 **結論**: p値が0.05未満のため、グループ間に「統計的に有意な差がある」と判断できます。")
                else:
                    st.warning("👉 **結論**: p値が0.05以上のため、グループ間に「有意な差があるとは言えない」と判断されます。")

            elif "二項分布" in dist:
                df['proportion'] = df[col_event] / df[col_total]
                formula = f"proportion ~ {exp_formula}"
                model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), var_weights=df[col_total]).fit()
                
                st.success("🎉 【正解】「上限のある割合データ」に対して二項分布（GLM）を適切に選択できました。")
                st.markdown("#### 📖 なぜこの選択が正しいのか？")
                st.info("割合データは「必ず0%〜100%の間に収まる」制約があり、確率が50%に近いほど分散が大きくなります。GLM（ロジットリンク関数）を使うことで、予測値が0未満や100%以上になることを防ぎ、分散の偏りも正しく処理できます。")
                st.write(model.summary())
                st.markdown("#### 🔍 解析結果の読み方")
                st.markdown("* **`coef` (偏回帰係数)**: プラスなら事象の発生を増やし、マイナスなら減らす働きがあります。\n* **`P>|z|` (p値)**: 0.05未満であれば「有意な影響がある（意味のある差である）」とみなします。")

            elif "ポアソン分布" in dist:
                formula = f"{Q(col_event)} ~ {exp_formula}"
                model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
                
                st.success("🎉 【正解】「上限のないカウントデータ」に対してポアソン分布（GLM）を適切に選択できました。")
                st.markdown("#### 📖 なぜこの選択が正しいのか？")
                st.info("「発生確率は低いが試行回数が多い（上限が不明な）事象」はポアソン分布に従います。データは「必ず0以上の整数」になり、平均値が大きくなるほど分散も大きくなる性質を正しくモデル化できます。")
                st.write(model.summary())
                st.markdown("#### 🔍 解析結果の読み方")
                st.markdown("* **`coef`**: GLM(ポアソン)では対数変換されているため「自然指数(e)の累乗」として影響を解釈します。\n* **`P>|z|`**: 0.05未満であれば統計的に意味のある要因です。")

            else:
                # OLS（正規分布）のトラップと正解判定
                if col_total != "なし":
                    df['proportion'] = df[col_event] / df[col_total]
                    formula_ols = f"proportion ~ {exp_formula}"
                    model = smf.ols(formula=formula_ols, data=df).fit()
                    
                    st.error("❌ 【不正解】変数の割り当ては合っていますが、モデル（分布）の選択が不適切です。")
                    st.warning("割合データに通常の線形回帰を当てはめると、予測値が0%未満になったり100%を超えたりする理論的破綻が起きます。二項分布（GLM）を選択してください。")
                    st.write(model.summary())
                    
                elif (df[col_event].dropna() % 1 == 0).all() and (df[col_event].dropna() >= 0).all() and df[col_event].max() < 100:
                    formula_ols = f"{Q(col_event)} ~ {exp_formula}"
                    model = smf.ols(formula=formula_ols, data=df).fit()
                    
                    st.error("❌ 【不正解】変数の割り当ては合っていますが、モデル（分布）の選択が不適切です。")
                    st.warning("カウントデータ（特に平均が小さいもの）に線形回帰を当てはめると、予測値がマイナスになる理論的破綻が起きます。ポアソン分布（GLM）を選択してください。")
                    st.write(model.summary())
                    
                else:
                    formula_ols = f"{Q(col_event)} ~ {exp_formula}"
                    model = smf.ols(formula=formula_ols, data=df).fit()
                    
                    st.success("🎉 【正解】連続データに対して「通常の線形回帰（正規分布）」を適切に選択できました！")
                    st.info("収量や重量などの「連続変数」は、正規分布を仮定するモデル（通常の線形回帰や分散分析など）で解析するのが基本です。")
                    st.write(model.summary())
                    st.markdown("#### 🔍 解析結果の読み方")
                    st.markdown("* **`coef`**: その変数が目的変数に与える「影響の向きと大きさ」です。\n* **`P>|t|`**: 0.05未満であれば「有意な影響がある」とみなします。")

        except Exception as e:
            st.error("🚨 解析中に予期せぬエラーが発生しました。データの構造や列名を確認してください。")
            st.code(str(e))
