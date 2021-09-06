import rospy

from core.Nodes import ActionController

rospy.init_node('test_acp')
acp = ActionController(node_id='acp', config_file='/tmp/test.json')

acp.run('Stop following me now')
