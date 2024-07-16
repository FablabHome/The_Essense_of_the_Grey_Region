datasets = [
    "/home/root_walker/workspace/ROS_projects/src/The_Essense_of_the_Grey_Region/action_controller/datasets/engineer_fair.yaml"
]

intent Greet {
    response = ["hi", "hoi", "hello", "hello, I am anchor", "Good day"]
    required_slots = []
}

intent Love {
    response = [
        "Give it up, we are impossible",
        "How can I do that with you?"
    ]
    required_slots = []
}
