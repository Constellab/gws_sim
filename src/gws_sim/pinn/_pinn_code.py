import pandas as pd
import deepxde as dde
import numpy as np
import sys
import os

print("étape a")

if len(sys.argv) != 7:
    print("Usage: python script.py <dataframe_file.csv> <string_list_file.txt> <t_start> <t_end> <initial_state>")
    sys.exit(1)
print("étape b")

dataframe_file = sys.argv[1]
print("étape c")

string_list_file_equations = sys.argv[2]
print("étape d")

string_list_file_params = sys.argv[3]
print("étape e")

t_start = sys.argv[4]
print("étape f")

t_end = sys.argv[5]
print("étape g")

string_list_file_initial_state = sys.argv[6]
print("étape h")


try:
    df = pd.read_csv(dataframe_file)
except FileNotFoundError:
    print(f"Error: DataFrame file '{dataframe_file}' not found.")
    sys.exit(1)

    # Read list of strings from text file
try:
    with open(string_list_file_equations, 'r') as f:
        string_list_equations = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: String list file '{string_list_file_equations}' not found.")
    sys.exit(1)

# Read list of strings from text file
try:
    with open(string_list_file_params, 'r') as f:
        string_list_params = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: String list file '{string_list_file_params}' not found.")
    sys.exit(1)

# Read list of strings from text file
try:
    with open(string_list_file_initial_state, 'r') as f:
        string_list_initial_state = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: String list file '{string_list_file_initial_state}' not found.")
    sys.exit(1)

C = []
ic = []
observe_y = []
dy_x = []
data_list = []

for i in range(len(string_list_params)):
    C[i] = dde.Variable(exec(string_list_params[i]))

external_trainable_variables = C

variable = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000, filename="variables.dat"
)

# Most backends
def ODE_model(x, y):

    equations = []

    for j in range(len(string_list_equations)):

        if j==len(string_list_equations): y[j] = y[:, j:]
        else: y[j] = y[:, j:j+1]

        dy_x[j] = dde.grad.jacobian(y, x, i=j)

        equations.append(dy_x[j] - exec(string_list_equations[j]))

    return equations

def boundary(_, on_initial):
    return on_initial

geom = dde.geometry.TimeDomain(t_start, t_end)

data_set_array = df.to_numpy()

observe_t = data_set_array[:, 0:1]
print("étape 0")

for k in range(len(string_list_equations)+1):

    # Initial conditions
    ic[k] = dde.icbc.IC(geom, lambda X: exec(string_list_file_initial_state[k]) , boundary, component=k)

    # Get the train data
    observe_y[k] = dde.icbc.PointSetBC(observe_t, data_set_array[:, k:k+1], component=k)
    print("étape 1")

data_list.append(ic)
data_list.append(observe_y)

data = dde.data.PDE(
    geom,
    ODE_model,
    data_list,
    num_domain=400,
    num_boundary=len(string_list_equations),
    anchors=observe_t,
)
print("étape 2")

net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
print("étape 3")
# train adam
model.compile(
    "adam", lr=0.001, external_trainable_variables=external_trainable_variables
)
losshistory, train_state = model.train(iterations=1, callbacks=[variable])
print("étape 4")

# train lbfgs
model.compile("L-BFGS", external_trainable_variables=external_trainable_variables)
losshistory, train_state = model.train(callbacks=[variable])

idx = np.argsort(train_state.X_test[:, 0])

t = train_state.X_test[idx, 0]
best_y = train_state.best_y

LIST = best_y[idx, :]

dft = pd.DataFrame(t, columns=['time'])
dfy = pd.DataFrame(LIST)

loss_steps = losshistory.steps
loss_train = np.array([np.sum(loss) for loss in losshistory.loss_train])
loss_test = np.array([np.sum(loss) for loss in losshistory.loss_test])

df_loss_steps = pd.DataFrame(loss_steps, columns=['loss_steps'])
df_loss_train = pd.DataFrame(loss_train, columns=['loss_train'])
df_loss_test = pd.DataFrame(loss_test, columns=['loss_test'])
print("étape 5")
df_result = pd.concat([dft, dfy, df_loss_steps, df_loss_test, df_loss_train], axis=1)
path = "pinn_result.csv"

print("étape6")
csv_file_path = os.path.join(os.path.abspath(
            os.path.dirname(__file__)),  "pinn_result.csv")
df_result.to_csv(csv_file_path, index=False)
print("étape 7")
