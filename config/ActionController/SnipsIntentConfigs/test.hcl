intent getWeather {
  allow_preempt = false
  required_slot location {
    confirm_intent = ConfirmWeather
    no_resp = "OK, but where do you want to know?"
    also_without time {
      no_resp = "OK, but when and where do you want to know?"
    }
  }
  required_slot time {
    confirm_intent = ConfirmWeather
    no_resp = "OK, but when do you want to know?"
  }
}

confirm_intent ConfirmWeather {
  max_re_ask = 3
}
