"""Premier League momentum dashboard — v4: hero chart + findings, results and caveats.
All model numbers are read from data/results.json (made by compute_results.py), never typed by hand."""
import json
from pathlib import Path
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

# ---- DATA ----
BASE = Path(__file__).parent
df_all = pd.read_csv(BASE / 'data' / 'epl_final.csv')
df_all['MatchDate'] = pd.to_datetime(df_all['MatchDate'])
R = json.load(open(BASE / 'data' / 'results.json'))
L, T = R['loso'], R['test']
CPS = ['15', '20', '30']

TOP3_BY_SEASON = {
    '2020/21': ['Man City', 'Man United', 'Liverpool'],
    '2021/22': ['Man City', 'Liverpool', 'Chelsea'],
    '2022/23': ['Man City', 'Arsenal', 'Man United'],
    '2023/24': ['Man City', 'Arsenal', 'Liverpool'],
    '2024/25': ['Liverpool', 'Arsenal', 'Man City'],
}
TEAM_COLORS = {'Man City': '#6CABDD', 'Man United': '#E63946', 'Liverpool': '#00D9A3',
               'Arsenal': '#EF0107', 'Chelsea': '#4CC9F0'}
BG, PANEL, TEXT, MUTED, ACCENT, LINE = '#0F1115', '#171A21', '#E8E8E8', '#8A8F98', '#00D9A3', '#262b33'
RED, GOLD = '#E63946', '#FFD166'
PARTIAL_NOTE = {'2024/25': "Data for this season ends in early May 2025 (35 of 38 matches per team), "
                           "so the last three matches are not shown."}


def build_team_sequence(df, team):
    rows = []
    for _, r in df.iterrows():
        if r['HomeTeam'] == team:
            res = 1 if r['FullTimeResult'] == 'H' else (0 if r['FullTimeResult'] == 'D' else -1)
        elif r['AwayTeam'] == team:
            res = 1 if r['FullTimeResult'] == 'A' else (0 if r['FullTimeResult'] == 'D' else -1)
        else:
            continue
        rows.append({'Date': r['MatchDate'], 'Result': res})
    seq = pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)
    level, last, opens, closes = 0.0, None, [], []
    for _, row in seq.iterrows():
        o = level
        if row['Result'] == 1: c = o + 1.0; last = 'green'
        elif row['Result'] == -1: c = o - 1.0; last = 'red'
        else:
            if last == 'green': c = max(o - o * 0.5, 0)
            elif last == 'red': c = min(o + (0 - o) * 0.5, 0)
            else: c = 0.0
        opens.append(o); closes.append(c); level = c
    seq['Open'], seq['Close'] = opens, closes
    seq['High'] = seq[['Open', 'Close']].max(axis=1)
    seq['Low'] = seq[['Open', 'Close']].min(axis=1)
    seq['EMA5'] = seq['Close'].ewm(span=5, adjust=False).mean()
    seq['MatchNum'] = range(1, len(seq) + 1)
    return seq


# ---- SMALL UI HELPERS ----
def label(text):
    return html.Div(text.upper(), style={'color': MUTED, 'fontSize': '11px', 'letterSpacing': '1px', 'marginBottom': '4px'})


def kpi_card(lbl, value, sub=""):
    return dbc.Card(dbc.CardBody([
        html.Div(lbl.upper(), style={'fontSize': '11px', 'color': MUTED, 'letterSpacing': '1px'}),
        html.Div(value, style={'fontSize': '26px', 'color': TEXT, 'fontWeight': '600'}),
        html.Div(sub, style={'fontSize': '12px', 'color': MUTED})]),
        style={'backgroundColor': PANEL, 'border': f'1px solid {LINE}'})


def panel(title, children, accent=None):
    return html.Div([html.Div(title.upper(), style={'color': accent or MUTED, 'fontSize': '11px',
                                                    'letterSpacing': '1px', 'fontWeight': '600', 'marginBottom': '8px'}),
                     *children],
                    style={'backgroundColor': PANEL, 'border': f'1px solid {LINE}', 'borderRadius': '8px',
                           'padding': '16px 18px', 'height': '100%'})


def p(text, **kw):
    return html.P(text, style={'color': TEXT, 'fontSize': '14px', 'lineHeight': '1.55', 'marginBottom': '8px', **kw})


def section_title(text, sub=None):
    return html.Div([html.H4(text, style={'color': TEXT, 'fontWeight': '700', 'marginBottom': '2px'}),
                     html.Div(sub or "", style={'color': MUTED, 'fontSize': '13px'})],
                    style={'margin': '38px 0 14px'})


# ---- DERIVED TEXT (all numbers from results.json) ----
pct = lambda x: f"{round(x * 100)}%"
m_test = [T[c]['model_correct'] for c in CPS]
b_test = [T[c]['baseline_pts_gd_correct'] for c in CPS]
found = (f"Held out 2025/26 and tested blind. The model got {m_test[0]}, {m_test[1]} and {m_test[2]} of the final top 5 "
         f"right at matchdays 15, 20 and 30. Simply ranking teams by points (goal difference breaking ties) got "
         f"{b_test[0]}, {b_test[1]} and {b_test[2]}. So far, momentum has not clearly added anything beyond the table.")

summary = dbc.Row([
    dbc.Col(panel("The question", [p("Does a team's early-season win/draw/loss momentum signal a top-5 finish?")]), md=3, className='mb-3'),
    dbc.Col(panel("What I did", [p(f"Turned every team's results into a momentum line, trained a logistic regression on "
                                   f"{R['meta']['n_train_seasons']} seasons (2000/01–2024/25), then tested it blind on 2025/26 "
                                   f"at matchdays 15, 20 and 30.")]), md=3, className='mb-3'),
    dbc.Col(panel("What I found", [p(found)], accent=ACCENT), md=3, className='mb-3'),
    dbc.Col(panel("What I don't know", [p("Whether momentum matters in a way my simple version can't see. "
                                          "One blind season is a very small sample. The live test on 2026/27 comes next.")],
                  accent=GOLD), md=3, className='mb-3'),
])

how_to_read = html.Div([
    html.Span("How to read the chart: ", style={'color': TEXT, 'fontWeight': '600'}),
    html.Span("a win adds 1 to momentum, a loss subtracts 1, and a draw eases it halfway back toward zero. "
              "In the deep-dive, green candles mean momentum rose and red means it fell (a draw can show as either). "
              "The dotted line is a 5-match smoothed average (EMA 5).", style={'color': MUTED})],
    style={'fontSize': '13px', 'marginBottom': '16px'})


# ---- MODEL TEST FIGURE + TABLE ----
def results_figure():
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Model', x=[f"Matchday {c}" for c in CPS], y=m_test, marker_color=ACCENT,
                         text=m_test, textposition='outside', textfont=dict(color=TEXT)))
    fig.add_trace(go.Bar(name='Points table (goal diff. breaks ties)', x=[f"Matchday {c}" for c in CPS], y=b_test,
                         marker_color='#5B6270', text=b_test, textposition='outside', textfont=dict(color=TEXT)))
    fig.update_layout(template='plotly_dark', plot_bgcolor=PANEL, paper_bgcolor=PANEL, font=dict(color=TEXT),
                      barmode='group', margin=dict(t=40, b=30, l=40, r=10), height=320,
                      title="2025/26 blind test: how many of the final top 5 were picked correctly",
                      title_font_size=14, yaxis=dict(range=[0, 5.8], dtick=1, title="Correct (out of 5)", gridcolor=LINE),
                      legend=dict(orientation='h', y=-0.18))
    return fig


def picks_cell(picks, actual):
    spans = []
    for t in picks:
        hit = t in actual
        spans.append(html.Span(t, style={'display': 'inline-block', 'margin': '2px 4px 2px 0', 'padding': '2px 8px',
                                         'borderRadius': '10px', 'fontSize': '12px',
                                         'backgroundColor': 'rgba(0,217,163,.16)' if hit else 'rgba(230,57,70,.16)',
                                         'color': ACCENT if hit else RED,
                                         'border': f"1px solid {ACCENT if hit else RED}"}))
    return html.Td(spans)


def results_table():
    actual = set(T['actual_top5'])
    th = lambda t: html.Th(t, style={'color': MUTED, 'fontSize': '11px', 'letterSpacing': '1px', 'fontWeight': '600',
                                     'borderBottom': f'1px solid {LINE}', 'padding': '6px 8px'})
    rows = []
    for c in CPS:
        rows.append(html.Tr([html.Td(f"MD {c}", style={'color': TEXT, 'fontWeight': '600', 'padding': '6px 8px'}),
                             picks_cell(T[c]['model'], actual), picks_cell(T[c]['baseline_pts_gd'], actual)],
                            style={'borderBottom': f'1px solid {LINE}'}))
    return html.Table([html.Thead(html.Tr([th(''), th('MODEL PICKS'), th('POINTS-TABLE PICKS')])), html.Tbody(rows)],
                      style={'width': '100%'})


actual_line = html.Div([html.Span("Actual final top 5: ", style={'color': MUTED}),
                        html.Span(", ".join(T['actual_top5']), style={'color': TEXT, 'fontWeight': '600'}),
                        html.Span("   Green = in the final top 5, red = not.", style={'color': MUTED})],
                       style={'fontSize': '12px', 'marginTop': '10px'})

# honest tie note: computed, not asserted
tie_cps = [c for c in CPS if T[c]['baseline_correct'] != T[c]['baseline_pts_gd_correct']]
tie_note = ""
if tie_cps:
    tie_note = (f"Note: at matchday {' and '.join(tie_cps)}, two teams were level on points at the 5th-place cut. "
                f"Breaking that tie by list order would have scored the table "
                f"{' and '.join(str(T[c]['baseline_correct']) for c in tie_cps)} instead of "
                f"{' and '.join(str(T[c]['baseline_pts_gd_correct']) for c in tie_cps)}, so I use goal difference, "
                f"the standard tie-break, to keep the comparison fair.")

loso_rows = [html.Tr([html.Td(f"MD {c}", style={'padding': '5px 8px', 'color': TEXT, 'fontWeight': '600'}),
                      html.Td(pct(L[c]['model_hit']), style={'color': ACCENT, 'padding': '5px 8px'}),
                      html.Td(pct(L[c]['naive_pts_gd_hit']), style={'color': TEXT, 'padding': '5px 8px'}),
                      html.Td(f"{L[c]['auc']:.2f}", style={'color': MUTED, 'padding': '5px 8px'})],
                     style={'borderBottom': f'1px solid {LINE}'}) for c in CPS]
loso_table = html.Table([html.Thead(html.Tr([html.Th(h, style={'color': MUTED, 'fontSize': '11px', 'letterSpacing': '1px',
                                                              'padding': '5px 8px', 'borderBottom': f'1px solid {LINE}'})
                                             for h in ['', 'MODEL', 'POINTS TABLE', 'AUC']])),
                         html.Tbody(loso_rows)], style={'width': '100%', 'fontSize': '13px'})

caveats = [
    "Every season here is viewed with hindsight. I knew the outcomes when I chose what to plot.",
    f"Only one blind test season so far (2025/26). Results from one season can be luck either way.",
    "Momentum is a deliberately simple win/draw/loss formula. It ignores opponent strength, home/away and injuries.",
    "The model is a first version. The mistakes it makes are what will guide the next features.",
    "Some training seasons have missing matches (2003/04, 2004/05, 2024/25), so their 'top 5' labels are approximate.",
    "This is a research project about patterns in football data. It is not betting advice.",
]

# ---- APP ----
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                title="Premier League Momentum", assets_folder=str(BASE / 'assets'))
server = app.server  # gunicorn target for deployment

app.layout = html.Div(style={'backgroundColor': BG, 'minHeight': '100vh', 'padding': '30px 5vw'}, children=[
    html.Div([
        html.H2("Does early momentum predict who finishes in the top 5?",
                style={'color': TEXT, 'fontWeight': '700', 'marginBottom': '4px'}),
        html.P("A Premier League data project — work in progress, shared openly with its limits.",
               style={'color': MUTED, 'fontSize': '14px'})], style={'marginBottom': '20px'}),

    summary,
    section_title("Explore the momentum", "Pick a season, then compare the top three or dive into one team."),
    how_to_read,

    dbc.Row([
        dbc.Col([label("Season"), dcc.Dropdown(id='season-dropdown',
                 options=[{'label': s, 'value': s} for s in TOP3_BY_SEASON], value='2021/22', clearable=False)], md=3),
        dbc.Col([label("Chart mode"), dcc.RadioItems(id='chart-mode', options=[
                 {'label': ' Top 3 Comparison', 'value': 'compare'}, {'label': ' Single Team Deep-Dive', 'value': 'deepdive'}],
                 value='compare', labelStyle={'display': 'block', 'color': TEXT, 'marginTop': '4px'},
                 inputStyle={'marginRight': '6px'})], md=4),
        dbc.Col(id='team-picker-container', md=3),
    ], style={'marginBottom': '12px'}),
    html.Div(id='season-note', style={'color': GOLD, 'fontSize': '12px', 'minHeight': '18px', 'marginBottom': '8px'}),
    dcc.Store(id='selected-team-store'),

    dcc.Graph(id='hero-chart', config={'displayModeBar': False}, style={'height': '480px', 'marginBottom': '25px'}),
    dbc.Row(id='kpi-row'),

    section_title("The blind test", "Models trained on 2000/01–2024/25, then scored on 2025/26, which they had never seen."),
    dbc.Row([
        dbc.Col(html.Div([dcc.Graph(figure=results_figure(), config={'displayModeBar': False})],
                         style={'backgroundColor': PANEL, 'border': f'1px solid {LINE}', 'borderRadius': '8px', 'padding': '6px'}),
                md=6, className='mb-3'),
        dbc.Col(panel("Who each approach picked", [results_table(), actual_line]), md=6, className='mb-3'),
    ]),
    html.Div(tie_note, style={'color': MUTED, 'fontSize': '12px', 'marginBottom': '14px'}),
    dbc.Row([
        dbc.Col(panel("Across 25 past seasons (each tested on a season it never trained on)", [
            loso_table,
            html.Div(f"Share of top-5 teams picked correctly. AUC is a 0–1 ranking score (0.5 = chance). "
                     f"Both approaches look similar here: the early table already tells most of the story.",
                     style={'color': MUTED, 'fontSize': '12px', 'marginTop': '8px'})]), md=6, className='mb-3'),
        dbc.Col(panel("What the misses show", [
            p(f"At matchday 20 the model left out {', '.join(T['20']['model_misses']) or 'nobody'} and picked "
              f"{', '.join(T['20']['model_false_alarms']) or 'nobody'} instead. It leans heavily on goal difference "
              f"and on how many goals a team has scored. I'm treating this as a clue about which features to add, not as a conclusion."),
            p("The smoothed momentum signal also changes direction between checkpoints, which is a sign it is noisy "
              "rather than a stable effect.", color=MUTED)]), md=6, className='mb-3'),
    ]),

    section_title("What this does not tell you"),
    panel("Caveats", [html.Ul([html.Li(c, style={'color': TEXT, 'fontSize': '14px', 'marginBottom': '6px'}) for c in caveats],
                              style={'paddingLeft': '18px', 'marginBottom': '0'})], accent=GOLD),

    html.Div([
        html.Div("Data: football-data.co.uk (2000/01–2024/25) and openfootball/england (2025/26).", style={'marginBottom': '4px'}),
        html.Div(["Built by Yubin · ", html.A("GitHub", href="https://github.com/ARSENE101", target="_blank",
                                              style={'color': ACCENT})]),
    ], style={'color': MUTED, 'fontSize': '12px', 'marginTop': '40px', 'paddingTop': '16px', 'borderTop': f'1px solid {LINE}'}),
])


@app.callback(Output('team-picker-container', 'children'),
              Input('chart-mode', 'value'), Input('season-dropdown', 'value'))
def show_team_picker(mode, season):
    if mode != 'deepdive':
        return []
    teams = TOP3_BY_SEASON[season]
    return [label("Team"), dcc.Dropdown(id='team-dropdown', options=[{'label': t, 'value': t} for t in teams],
                                        value=teams[0], clearable=False)]


@app.callback(Output('selected-team-store', 'data'), Input('team-dropdown', 'value'))
def store_selected_team(v):
    return v


@app.callback(Output('season-note', 'children'), Input('season-dropdown', 'value'))
def season_note(season):
    return PARTIAL_NOTE.get(season, "")


@app.callback(Output('hero-chart', 'figure'), Output('kpi-row', 'children'),
              Input('season-dropdown', 'value'), Input('chart-mode', 'value'), Input('selected-team-store', 'data'))
def update_hero(season, mode, stored_team):
    sdf = df_all[df_all['Season'] == season]
    fig = go.Figure()
    if mode == 'compare':
        teams = TOP3_BY_SEASON[season]; finals = {}
        for t in teams:
            seq = build_team_sequence(sdf, t); finals[t] = seq['EMA5'].iloc[-1]
            fig.add_trace(go.Scatter(x=seq['MatchNum'], y=seq['EMA5'], mode='lines', name=t,
                                     line=dict(color=TEAM_COLORS.get(t, '#888'), width=3)))
        ordered = sorted(finals.values(), reverse=True)
        leader = max(finals, key=finals.get)
        kpis = [dbc.Col(kpi_card("Highest final momentum", leader, f"EMA-5 = {round(finals[leader], 2)}"), md=4, className='mb-2'),
                dbc.Col(kpi_card("Gap to next", f"{round(ordered[0] - ordered[1], 2)}", "momentum points"), md=4, className='mb-2'),
                dbc.Col(kpi_card("Teams tracked", str(len(teams)), ", ".join(teams)), md=4, className='mb-2')]
        title = f"Top 3 momentum, EMA(5) | {season}"
    else:
        teams = TOP3_BY_SEASON[season]
        team = stored_team if stored_team in teams else teams[0]
        seq = build_team_sequence(sdf, team)
        fig.add_trace(go.Candlestick(x=seq['MatchNum'], open=seq['Open'], high=seq['High'], low=seq['Low'],
                                     close=seq['Close'], increasing_line_color=ACCENT, decreasing_line_color=RED,
                                     name=team, showlegend=False))
        fig.add_trace(go.Scatter(x=seq['MatchNum'], y=seq['EMA5'], mode='lines', name='EMA(5)',
                                 line=dict(color=GOLD, width=2, dash='dot')))
        kpis = [dbc.Col(kpi_card("Team", team, f"{len(seq)} matches"), md=4, className='mb-2'),
                dbc.Col(kpi_card("Peak EMA(5)", round(seq['EMA5'].max(), 2), "highest momentum reached"), md=4, className='mb-2'),
                dbc.Col(kpi_card("Final EMA(5)", round(seq['EMA5'].iloc[-1], 2), "momentum at last match shown"), md=4, className='mb-2')]
        title = f"{team} momentum deep-dive | {season}"
        fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(template='plotly_dark', plot_bgcolor=PANEL, paper_bgcolor=PANEL, font=dict(color=TEXT),
                      title=title, xaxis_title="Match number", yaxis_title="Momentum", margin=dict(t=50))
    return fig, kpis


if __name__ == '__main__':
    app.run(debug=True)
