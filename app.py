"""Premier League momentum dashboard — v4: hero chart + findings, results and caveats.
All model numbers are read from data/results.json (made by compute_results.py), never typed by hand."""
import json
from pathlib import Path
import dash
from dash import dcc, html, Input, Output
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
TEAM_COLORS = {'Man City': '#6CABDD', 'Man United': '#FF9F1C', 'Liverpool': '#C6F36B',
               'Arsenal': '#FF5A5F', 'Chelsea': '#8E9BFF'}
TEXT, MUTED, ACCENT = '#EEF3EA', '#8DA397', '#C6F36B'
RED, GOLD = '#FF6B6B', '#FFD166'
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



# ---- DERIVED TEXT (all numbers from results.json) ----
pct = lambda x: f"{round(x * 100)}%"
m_test = [T[c]['model_correct'] for c in CPS]
b_test = [T[c]['baseline_pts_gd_correct'] for c in CPS]
slash = lambda v: "/".join(str(x) for x in v)
lead = (f"Tested blind on 2025/26, the model got {slash(m_test)} of the final top 5 right at matchdays 15, 20 and 30. "
        f"The plain points table got {slash(b_test)}.")


def div(cls, *children, **kw):
    return html.Div(list(children), className=cls, **kw)


def pills(id_, options, value):
    return dcc.RadioItems(id=id_, options=[{'label': o, 'value': v} for o, v in options], value=value,
                          className='pills', inputClassName='pill-input')


def stat(n, l):
    return div('stat', div('n', str(n)), div('l', l))


# ---- BLIND TEST AS FIXTURES ----
def chips(picks, actual):
    return [html.Span(t, className='chip ' + ('hit' if t in actual else 'miss')) for t in picks]


def fixture(c):
    actual = set(T['actual_top5'])
    m, b = T[c]['model_correct'], T[c]['baseline_pts_gd_correct']
    verdict, vcls = ("Level", '') if m == b else ((f"Model +{m - b}", 'model') if m > b else (f"Table +{b - m}", ''))
    return div('fixture',
               div('fx-top', div('fx-md', f"Matchday {c}"),
                   div('fx-score',
                       div('side', div('num' + (' win' if m > b else ''), str(m)), div('nm', 'Model')),
                       div('sep', '–'),
                       div('side', div('num' + (' win' if b > m else ''), str(b)), div('nm', 'Points table'))),
                   div(f'fx-verdict {vcls}', verdict)),
               div('chiprow', html.Span('Model picked', className='who'), *chips(T[c]['model'], actual)),
               div('chiprow', html.Span('Table picked', className='who'), *chips(T[c]['baseline_pts_gd'], actual)))


tie_cps = [c for c in CPS if T[c]['baseline_correct'] != T[c]['baseline_pts_gd_correct']]
tie_note = ""
if tie_cps:
    tie_note = (f"Why goal difference? At matchday {' and '.join(tie_cps)}, two teams were level on points at the 5th-place cut. "
                f"Breaking that tie by list order would have scored the table "
                f"{' and '.join(str(T[c]['baseline_correct']) for c in tie_cps)} instead of "
                f"{' and '.join(str(T[c]['baseline_pts_gd_correct']) for c in tie_cps)}. I use goal difference, the standard tie-break, to keep it fair.")

loso_rows = [html.Tr([html.Td(f"MD {c}"), html.Td(pct(L[c]['model_hit']), className='m'),
                      html.Td(pct(L[c]['naive_pts_gd_hit'])), html.Td(f"{L[c]['auc']:.2f}")]) for c in CPS]
loso_table = html.Table([html.Thead(html.Tr([html.Th(h) for h in ['', 'Model', 'Points table', 'AUC']])),
                         html.Tbody(loso_rows)], className='t')

caveats = [
    "Every season here is viewed with hindsight. I knew the outcomes when I chose what to plot.",
    "Only one blind test season so far (2025/26). One season can be luck either way.",
    "Momentum is a deliberately simple win/draw/loss formula. It ignores opponent strength, home/away and injuries.",
    "This is a first version. Its mistakes will guide the next features.",
    "Some training seasons have missing matches (2003/04, 2004/05, 2024/25), so their top-5 labels are approximate.",
    "This is research into patterns in football data. It is not betting advice.",
]

# ---- APP ----
app = dash.Dash(__name__, suppress_callback_exceptions=True, title="Premier League Momentum",
                assets_folder=str(BASE / 'assets'))
server = app.server  # gunicorn target for deployment

app.layout = div('wrap',
    div('top', html.Span("Momentum vs the table", className='brand'),
        html.Span([html.B("Work in progress"), " · tested on 2025/26"], className='status')),
    html.H1("Does early momentum predict a top-5 finish?"),
    html.P(lead, className='lead'),
    html.P(f"A Premier League project. I turned results into a momentum line, trained a logistic regression on "
           f"{R['meta']['n_train_seasons']} seasons (2000/01 to 2024/25), then tested it on a season it had never seen. "
           f"Signal, not proof.", className='sub'),

    div('hero',
        div('hero-head',
            div('group', div('group-label', "Season"),
                pills('season-dropdown', [(s, s) for s in TOP3_BY_SEASON], '2021/22')),
            div('group', div('group-label', "View"),
                pills('chart-mode', [("Top 3 compared", 'compare'), ("One team in detail", 'deepdive')], 'compare')),
            html.Div(id='team-picker-container', className='group')),
        html.Div(id='season-note', className='season-note'),
        dcc.Store(id='selected-team-store'),
        dcc.Graph(id='hero-chart', config={'displayModeBar': False}, className='hero-chart'),
        html.Div(id='kpi-row', className='scoreboard'),
        div('howto', html.B("How to read it: "),
            "a win adds 1 to momentum, a loss subtracts 1, and a draw eases it halfway back to zero. In the detailed view, "
            "green candles mean momentum rose and red means it fell (a draw can show as either). The dotted line is a "
            "5-match smoothed average (EMA 5).")),

    html.H2("The blind test"),
    html.P("Both approaches picked a top 5 using only results up to that matchday. The score is how many of the final top 5 they got right.",
           className='h2-sub'),
    div('cols',
        div('panel', *[fixture(c) for c in CPS],
            div('key', html.Span("Actual final top 5: ", className='muted'), html.B(", ".join(T['actual_top5'])),
                html.Span(".  Green chips made the top 5, red did not.", className='muted'))),
        div('panel', html.H3("Across 25 past seasons"), loso_table,
            html.P("Share of top-5 teams picked correctly, with each season tested while held out of training. AUC is a 0 to 1 "
                   "ranking score (0.5 is chance). The two approaches look alike: the early table already tells most of the story.",
                   className='muted', style={'fontSize': '13px', 'marginTop': '10px'}))),
    div('tie', tie_note),

    div('cols-even',
        div('panel', html.H3("What the misses show"),
            html.P(f"At matchday 20 the model left out {', '.join(T['20']['model_misses']) or 'nobody'} and picked "
                   f"{', '.join(T['20']['model_false_alarms']) or 'nobody'} instead. It leans heavily on goal difference and goals scored. "
                   f"I treat this as a clue about which features to add, not a conclusion."),
            html.P("The smoothed momentum signal also changes direction between checkpoints, which suggests noise rather than a stable effect.",
                   className='muted')),
        div('panel caveats', html.H3("What this does not tell you"),
            html.Ul([html.Li(c) for c in caveats]))),

    div('foot', html.Div("Data: football-data.co.uk (2000/01 to 2024/25) and openfootball/england (2025/26)."),
        html.Div(["Built by Yubin. ", html.A("GitHub", href="https://github.com/ARSENE101", target="_blank")], style={'marginTop': '4px'})),
)


@app.callback(Output('team-picker-container', 'children'),
              Input('chart-mode', 'value'), Input('season-dropdown', 'value'))
def show_team_picker(mode, season):
    if mode != 'deepdive':
        return []
    teams = TOP3_BY_SEASON[season]
    return [div('group-label', "Team"), pills('team-dropdown', [(t, t) for t in teams], teams[0])]


@app.callback(Output('selected-team-store', 'data'), Input('team-dropdown', 'value'))
def store_selected_team(v):
    return v


@app.callback(Output('season-note', 'children'), Input('season-dropdown', 'value'))
def season_note(season):
    return PARTIAL_NOTE.get(season, "")


def style_fig(fig, title):
    fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(family="Barlow, sans-serif", color=TEXT, size=14),
                      title=dict(text=title, font=dict(family="Barlow Condensed, sans-serif", size=22), x=0, xanchor='left'),
                      margin=dict(t=48, b=44, l=48, r=10), legend=dict(orientation='h', y=1.08, x=1, xanchor='right'),
                      xaxis=dict(title="Match number", gridcolor='rgba(255,255,255,.07)', zeroline=False),
                      yaxis=dict(title="Momentum", gridcolor='rgba(255,255,255,.07)', zerolinecolor='rgba(255,255,255,.2)'))
    return fig


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
                                     line=dict(color=TEAM_COLORS.get(t, '#888'), width=3.5)))
        ordered = sorted(finals.values(), reverse=True)
        leader = max(finals, key=finals.get)
        kpis = [stat(leader, f"Highest final momentum (EMA {round(finals[leader], 2)})"),
                stat(f"{round(ordered[0] - ordered[1], 2)}", "Gap to the next team"),
                stat(len(teams), ", ".join(teams))]
        title = f"Momentum of the top 3, {season}"
    else:
        teams = TOP3_BY_SEASON[season]
        team = stored_team if stored_team in teams else teams[0]
        seq = build_team_sequence(sdf, team)
        fig.add_trace(go.Candlestick(x=seq['MatchNum'], open=seq['Open'], high=seq['High'], low=seq['Low'],
                                     close=seq['Close'], increasing_line_color=ACCENT, decreasing_line_color=RED,
                                     name=team, showlegend=False))
        fig.add_trace(go.Scatter(x=seq['MatchNum'], y=seq['EMA5'], mode='lines', name='EMA 5',
                                 line=dict(color=GOLD, width=2.5, dash='dot')))
        kpis = [stat(team, f"{len(seq)} matches"), stat(round(seq['EMA5'].max(), 2), "Peak EMA 5"),
                stat(round(seq['EMA5'].iloc[-1], 2), "EMA 5 at the last match shown")]
        title = f"{team}, match by match, {season}"
        fig.update_xaxes(rangeslider_visible=False)
    return style_fig(fig, title), kpis


if __name__ == '__main__':
    app.run(debug=True)
