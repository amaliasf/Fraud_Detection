import pandas as pd
import plotly.graph_objects as go
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output

model=joblib.load('dt_fraud_up')
#fraud_train = pd.read_csv("fraudTrain.csv")
fraud_test = pd.read_csv("fraud_test.csv")

# fraud_train['trans_date_trans_time']=pd.to_datetime(fraud_train['trans_date_trans_time'])
# fraud_train['trans_date']=fraud_train['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
# fraud_train['trans_date']=pd.to_datetime(fraud_train['trans_date'])
# fraud_train['dob']=pd.to_datetime(fraud_train['dob'])

fraud_test['trans_date_trans_time']=pd.to_datetime(fraud_test['trans_date_trans_time'])
fraud_test['trans_date']=fraud_test['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
fraud_test['trans_date']=pd.to_datetime(fraud_test['trans_date'])
fraud_test['dob']=pd.to_datetime(fraud_test['dob'])

#fraud_train.drop("Unnamed: 0",axis=1,inplace=True)
fraud_test.drop("Unnamed: 0",axis=1,inplace=True)

#fraud_train["is_fraud"]=fraud_train.is_fraud.apply(lambda x: "Yes" if x==1 else "No")
fraud_test["is_fraud"]=fraud_test.is_fraud.apply(lambda x: "Yes" if x==1 else "No")

#fraud_train["is_fraud"].astype("object")
fraud_test["is_fraud"].astype("object")

# fraud_train["age"] = fraud_train["trans_date"]- fraud_train["dob"]
# fraud_train["age"]= fraud_train["age"].astype('timedelta64[Y]')

fraud_test["age"] = fraud_test["trans_date"]- fraud_test["dob"]
fraud_test["age"]= fraud_test["age"].astype('timedelta64[Y]')

# fraud_train['trans_month'] = fraud_train['trans_date'].dt.month
# fraud_train['trans_year'] = fraud_train['trans_date'].dt.year

fraud_test['trans_month'] = fraud_test['trans_date'].dt.month
fraud_test['trans_year'] = fraud_test['trans_date'].dt.year

# fraud_train['latitudinal_distance'] = abs(round(fraud_train['merch_lat']-fraud_train['lat'],3))
# fraud_train['longitudinal_distance'] = abs(round(fraud_train['merch_long']-fraud_train['long'],3))

fraud_test['latitudinal_distance'] = abs(round(fraud_test['merch_lat']-fraud_test['lat'],3))
fraud_test['longitudinal_distance'] = abs(round(fraud_test['merch_long']-fraud_test['long'],3))

drop_cols = ['trans_date_trans_time','trans_num','city','lat','long','job','dob','merch_lat','merch_long','trans_date','state']
# fraud_train=fraud_train.drop(drop_cols,axis=1)
fraud_test=fraud_test.drop(drop_cols,axis=1)


# fraud_train.drop(['merchant','first','last','street','zip','unix_time'],axis=1,inplace=True)
fraud_test.drop(['merchant','first','last','street','zip','unix_time'],axis=1,inplace=True)

cat_cols = ['category','gender']

# fraud_train[cat_cols] = fraud_train[cat_cols].astype("category")
# fraud_train['is_fraud'] = fraud_train['is_fraud'].astype("category")
fraud_test[cat_cols] = fraud_test[cat_cols].astype("category")
fraud_test['is_fraud'] = fraud_test['is_fraud'].astype("category")

# X_train = fraud_train.drop('is_fraud', axis=1)
# y_train = fraud_train['is_fraud']

X_test = fraud_test.drop('is_fraud', axis=1)
y_test = fraud_test['is_fraud']


encoder = OneHotEncoder(drop='first', sparse=False)
encoder.fit(X_test[cat_cols])

# X_train_encoded = pd.DataFrame(encoder.transform(X_train[cat_cols]),
#                                index=X_train.index,
#                                columns=encoder.get_feature_names(X_train[cat_cols].columns))
# X_train_dummy = pd.concat([X_train.select_dtypes(exclude='category'),
#                            X_train_encoded], axis=1)

X_test_encoded = pd.DataFrame(encoder.transform(X_test[cat_cols]),
                               index=X_test.index,
                               columns=encoder.get_feature_names(X_test[cat_cols].columns))
X_test_dummy = pd.concat([X_test.select_dtypes(exclude='category'),
                           X_test_encoded], axis=1)

# input prediction
baris=0

# Card Contents
y_pred_dt_test_up = model.predict(X_test_dummy)
acc_score = ((accuracy_score(y_true=y_test, y_pred=y_pred_dt_test_up))*100).round(2)
pred_score = ((precision_score(y_true=y_test, y_pred=y_pred_dt_test_up, pos_label='Yes'))*100).round(2)
recall = ((recall_score(y_true=y_test, y_pred=y_pred_dt_test_up, pos_label='Yes'))*100).round(2)


card_accuracy = [
    dbc.CardHeader("Accuracy Score"),
    dbc.CardBody(
        [
            html.H1(
                f"{acc_score}%",
                className="card-text",
            ),
        ]
    ),
]

card_precision = [
    dbc.CardHeader("Precision Score"),
    dbc.CardBody(
        [
            html.H1(
                f"{pred_score}%",
                className="card-text",
            ),
        ]
    ),
]

card_recall = [
    dbc.CardHeader("Recall Score"),
    dbc.CardBody(
        [
            html.H1(
                f"{recall}%",
                className="card-text",
            ),
        ]
    ),
]

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
app.title = 'Fraud Detection Dashboard'
server = app.server

app.layout = html.Div(children=[
    dbc.NavbarSimple(
        children=[html.H1('Fraud Detection Dashboard'),
        ],
        brand="",
        brand_href="#",
        color="#6286c4",
    ),
    html.Br(),
    dbc.Row(
        dbc.Alert(id='tbl_out', color='dark'),
    ),
    dbc.Row([
        dbc.Col([
            html.Br(), html.Br(),
            dbc.Row(
                dash_table.DataTable(
                id='table_data_test',
                columns=[{'name': i, 'id': i} for i in X_test.columns],
                data=X_test.to_dict('records'),
                fixed_rows={'headers': True},
                style_cell={
                    'minWidth':80, 'maxWidth': 300, 'width': 110
                },
                style_header={
                    'backgroundColor': '#6286c4',
                    'color': 'black'
                },
                style_data={
                    'backgroundColor': 'darkgray',
                    'color': 'black'
                }

        ),
            )
        ], width=6),
        # dbc.Col(dcc.Graph(id='prediction'), width=6), 
        dbc.Col(
            [
                dbc.Row(
                    dcc.Graph(id='prediction',
                            figure=go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = model.predict_proba(X_test_dummy.iloc[[baris],:])[0][1],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Probability"},
                            gauge = {'axis': {'range': [0,1]},
                                    'bar': {'color': '#6286c4'},
                                    'borderwidth': 2,
                                    'bordercolor': 'black',
                                    'steps': [
                                        {'range': [0, 0.5], 'color':'darkgray'},
                                        {'range': [0.5, 1], 'color': 'gray'}
                                    ],
                                    'threshold': {
                                        'line': {'color': 'red', 'width': 2},
                                        'thickness': 0.9,
                                        'value': 0.5
                                    }
                                    
                                    }
                        )),
            )
                ),
                dbc.Row([
                    dbc.Col(dbc.Card(card_accuracy, color="#6286c4", inverse=True)),
                    dbc.Col(dbc.Card(card_precision, color="#6286c4", inverse=True)),
                    dbc.Col(dbc.Card(card_recall, color="#6286c4", inverse=True)),
                ]),
                html.Br(),
                # dbc.Row(
                #     dbc.Alert(id='tbl_out', color='dark'),
                # )
            ], width=6), 
])
])

@app.callback(
    Output('tbl_out', 'children'),
    Input('table_data_test', 'active_cell')
    )

def update_row(active_cell):
    if active_cell:
        row_selectable='single'
        cc_num = X_test.iloc[active_cell['row'],0]
        prob = ((model.predict_proba(X_test_dummy.iloc[[active_cell['row']],:])[0][1])*100).round(1)
        return f"The transaction (CC Number : {cc_num}) has probability of {prob}% to being a Fraud"

    else:
        return "Please select the row you want to predict."
    # return str(active_cell['row']) if active_cell else "Click the table"

@app.callback(
    Output('prediction', 'figure'),
    Input('table_data_test', 'active_cell')
    )

def update_graphs(active_cell):
    row_baris = int(active_cell['row'])
    return go.Figure(go.Indicator(
                mode = "gauge+number",
                value = model.predict_proba(X_test_dummy.iloc[[row_baris],:])[0][1],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability"},
                gauge = {'axis': {'range': [0,1]},
                        'bar': {'color': '#6286c4'},
                        'borderwidth': 2,
                        'bordercolor': 'black',
                        'steps': [
                            {'range': [0, 0.5], 'color':'darkgray'},
                            {'range': [0.5, 1], 'color': 'gray'}
                        ],
                        'threshold': {
                            'line': {'color': 'red', 'width': 2},
                            'thickness': 0.9,
                            'value': 0.5
                        }
                        
                        }
            ))
    
    
    

if __name__ == "__main__":
    app.run_server()