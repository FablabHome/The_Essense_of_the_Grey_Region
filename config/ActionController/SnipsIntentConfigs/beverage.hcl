engines = [
    "/home/root_walker/workspace/ROS_projects/src/The_Essense_of_the_Grey_Region/action_controller/engines/beverage"
]

intent prepareBeverage {
    required_slots = ["beverage_type", "number_of_cups"]
    max_re_ask = 3
    confirm_intent = ""
    confirm_responses = {
        "beverage_type" = "Can you tell me what do you want to order?"
        "number_of_cups" = "Can you tell me how much you'd like to order?"
        "number_of_cups beverage_type" = "Can you tell me what do you want to order and how much?"
    }
}
