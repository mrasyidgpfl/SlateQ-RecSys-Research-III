import gin
from absl import app, flags
from recsim_ng.core import value, variable
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf
from recsim_ng.core.network import Network as TFNetwork

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Gin files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings.')

@gin.configurable
def simple_story():
    state = variable.Variable(
        name='state',
        spec=value.ValueSpec(state=value.FieldSpec())  # Use plain FieldSpec
    )
    state.initial_value = variable.value(lambda: value.Value(state=tf.constant(0.0, dtype=tf.float32)))
    state.value = variable.value(lambda prev: value.Value(state=prev.get('state') + tf.constant(1.0, dtype=tf.float32)), (state.previous,))
    return [state]

def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)
    story = simple_story()
    network = TFNetwork(variables=story)
    rt = runtime.TFRuntime(network=network)
    trajectory = rt.trajectory(length=5)
    state_value = trajectory.get('state')  # Get the Value object
    if state_value:
        # Access the 'state' field directly
        state_data = state_value.get('state')
        if isinstance(state_data, tf.Tensor):
            for value in state_data.numpy():
                print(value)
        else:
            print(state_data)
    else:
        print("No state in trajectory")

if __name__ == '__main__':
    app.run(main)