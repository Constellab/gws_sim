import pandas as pd
import deepxde as dde
import numpy as np
import sys
import os

print("step a")

if len(sys.argv) != 10:
    raise Exception("Usage: python script.py <dataframe_file.csv> <string_list_file.txt> <t_start> <t_end> <initial_state>")

dataframe_file = sys.argv[1]
string_list_file_equations = sys.argv[2]
string_list_file_params = sys.argv[3]
t_start = sys.argv[4]
t_end = sys.argv[5]
string_list_file_initial_state = sys.argv[6]
number_hidden_layers = sys.argv[7]
width_hidden_layers = sys.argv[8]
number_iterations = sys.argv[9]

if not os.path.exists(dataframe_file):
    raise Exception(f"Error: DataFrame file '{dataframe_file}' not found.")

try:
    df = pd.read_csv(dataframe_file)
except Exception as err:
    raise Exception(f"Error: DataFrame file '{dataframe_file}' not found.\n{err}")

    # Read list of strings from text file
try:
    with open(string_list_file_equations, 'r') as f:
        string_list_equations = [line.strip() for line in f.readlines()]
except FileNotFoundError as err:
    raise Exception(f"Error: String list file '{string_list_file_equations}' not found.\n{err}")

# Read list of strings from text file
try:
    with open(string_list_file_params, 'r') as f:
        string_list_params = [line.strip() for line in f.readlines()]
except FileNotFoundError as err:
    raise Exception(f"Error: String list file '{string_list_file_params}' not found.\n{err}")

# Read list of strings from text file
try:
    with open(string_list_file_initial_state, 'r') as f:
        string_list_initial_state = [line.strip() for line in f.readlines()]
except FileNotFoundError as err:
    raise Exception(f"Error: String list file '{string_list_file_initial_state}' not found.\n{err}")

C = []
ic = []
observe_y = []
data_list = []

print(f"type of string_list_equations : {type(string_list_equations)}, number of elements = {len(string_list_equations)}")
print(string_list_equations)
print(f"type of string_list_params : {type(string_list_params)}")
print(string_list_params)
print(f"type of string_list_initial_state : {type(string_list_initial_state)}")
print(string_list_initial_state)
print(f"type of string_list_initial_state[0] : {type(string_list_initial_state[0])}")
print(f"type of float(string_list_initial_state[0]) : {type(float(string_list_initial_state[0]))}")

for i in range(len(string_list_params)):
    C.append(dde.Variable(float(string_list_params[i])))
    print(C)

external_trainable_variables = C

variable = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000 #, filename="variables.dat"
)

# Most backends
def ODE_model(x, y_):

    '''D = 0 # Batch
    S_in = 100

    y1, y2, y3 = y_[:, 0:1], y_[:, 1:2], y_[:, 2:3]  # X, S, P

    dy1_x = dde.grad.jacobian(y_, x, i=0)
    dy2_x = dde.grad.jacobian(y_, x, i=1)
    dy3_x = dde.grad.jacobian(y_, x, i=2)

    return [dy1_x - y1,  dy2_x - y2, dy3_x - y3]'''

    equations_list = []
    dy_x = []
    y = []

    for j in range(len(string_list_equations)):
        y.append(y_[:, j:j+1])
        dy_x.append(dde.grad.jacobian(y_, x, i=j))
        #print(f"dy_x = {dy_x}")

    for i in range(len(string_list_equations)):
        equations_list.append(dy_x[i] - eval(string_list_equations[i]))

    return equations_list

def boundary(_, on_initial):
    return on_initial

geom = dde.geometry.TimeDomain(float(t_start), float(t_end))

df_data = df
data_set_array = df.to_numpy()

observe_t = data_set_array[:, 0:1]
print(observe_t)
for k in range(len(string_list_equations)):

    # Initial conditions
    ic.append(dde.icbc.IC(geom, lambda X: float(string_list_initial_state[k]), boundary, component=k))
    print(f"float(string_list_initial_state[{k}]) = {float(string_list_initial_state[k])}")
    # Get the train data
    observe_y.append(dde.icbc.PointSetBC(observe_t, data_set_array[:, k+1:k+2], component=k))
    print(data_set_array[:, k+1:k+2])
    print("step 1")

data_list = ic + observe_y
print(data_list)
data = dde.data.PDE(
    geom,
    ODE_model,
    data_list,
    num_domain=400,
    num_boundary=len(string_list_equations),
    anchors=observe_t,
)
print("step 2")

net = dde.nn.FNN([1] + [int(width_hidden_layers)] * int(number_hidden_layers) + [len(string_list_equations)], "tanh", "Glorot uniform")
model = dde.Model(data, net)
print("step 3")

# train adam
model.compile(
    "adam", lr=0.001, external_trainable_variables=external_trainable_variables
)
losshistory, train_state = model.train(iterations=int(number_iterations), callbacks=[variable])
print("step 4")

# train lbfgs
model.compile("L-BFGS", external_trainable_variables=external_trainable_variables)
losshistory, train_state = model.train(callbacks=[variable])

idx = np.argsort(train_state.X_test[:, 0])

t = train_state.X_test[idx, 0]
best_y = train_state.best_y

LIST = best_y[idx, :]

dft = pd.DataFrame(t, columns=['time'])
dfy = pd.DataFrame(LIST)

new_column_names_result = [f'result_col_{i+1}' for i in range(len(dfy.columns))]
dfy.columns = new_column_names_result

new_column_names_data = [f'data_col_{i+1}' for i in range(len(df_data.columns))]
df_data.columns = new_column_names_data

loss_steps = losshistory.steps
loss_train = np.array([np.sum(loss) for loss in losshistory.loss_train])
loss_test = np.array([np.sum(loss) for loss in losshistory.loss_test])

df_loss_steps = pd.DataFrame(loss_steps, columns=['loss_steps'])
df_loss_train = pd.DataFrame(loss_train, columns=['loss_train'])
df_loss_test = pd.DataFrame(loss_test, columns=['loss_test'])

print("step 5")
df_result = pd.concat([dft, dfy, df_data, df_loss_steps, df_loss_test, df_loss_train], axis=1)
path = "pinn_result.csv"

print("step 6")
csv_file_path = os.path.join(os.path.abspath(
            os.path.dirname(__file__)),  "pinn_result.csv")
df_result.to_csv(csv_file_path, index=False)
print("step 7")
