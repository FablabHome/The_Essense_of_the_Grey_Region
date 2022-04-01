
intent turnLightOn {
    allow_preempt = false
    max_re_ask = 3
    confirm_intent = "ConfirmRoomLightOn"
    required_slots = ["room"]
}

intent turnLightOff {
    allow_preempt = false
    max_re_ask = 3
    confirm_intent = "ConfirmRoomLightOff"
    required_slots = ["room"]
}

intent setLightsColor {
    allow_preempt = false
    max_re_ask = 3
    confirm_intent = "ConfirmColor"
    required_slots = ["color"]
}



confirm_intent ConfirmRoomLightOn {
    echo = off
}

confirm_intent ConfirmRoomLightOff {
    echo = off
}

confirm_intent ConfirmColor {
    echo = off
}
