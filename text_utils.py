def number_to_words_id(number):
    """Convert number to Indonesian words
    
    Args:
        number (int/float/str): Number to convert (can be integer, float, or string)
        
    Returns:
        str: Number in Indonesian words
    """
    # Handle string input and convert to int/float
    if isinstance(number, str):
        try:
            if '.' in number:
                number = float(number)
            else:
                number = int(number)
        except ValueError:
            return number  # Return as is if not a number
    
    # Handle float numbers
    if isinstance(number, float):
        integer_part = int(number)
        decimal_part = int(round((number - integer_part) * 100))  # Get 2 decimal places
        
        words = []
        if integer_part != 0:
            words.append(number_to_words_id(integer_part))
        
        if decimal_part != 0:
            if words:  # If there's an integer part
                words.append("koma")
            words.append(number_to_words_id(decimal_part))
            
        return ' '.join(words)
    
    # Handle negative numbers
    if number < 0:
        return "minus " + number_to_words_id(abs(number))
    
    # Handle zero
    if number == 0:
        return "nol"
    
    # Indonesian number words
    units = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
    teens = ["sepuluh", "sebelas", "dua belas", "tiga belas", "empat belas", 
             "lima belas", "enam belas", "tujuh belas", "delapan belas", "sembilan belas"]
    tens = ["", "sepuluh", "dua puluh", "tiga puluh", "empat puluh", 
            "lima puluh", "enam puluh", "tujuh puluh", "delapan puluh", "sembilan puluh"]
    
    # Handle numbers less than 1000
    def convert_less_than_thousand(n):
        if n < 10:
            return units[n]
        elif 10 <= n < 20:
            return teens[n - 10]
        elif n < 100:
            ten, unit = divmod(n, 10)
            if unit == 0:
                return tens[ten]
            else:
                return f"{tens[ten]} {units[unit]}"
        else:
            hundred, remainder = divmod(n, 100)
            if remainder == 0:
                return "seratus" if hundred == 1 else f"{units[hundred]} ratus"
            else:
                hundred_text = "seratus" if hundred == 1 else f"{units[hundred]} ratus"
                return f"{hundred_text} {convert_less_than_thousand(remainder)}"
    
    # Handle numbers >= 1000
    scales = ["", "ribu", "juta", "miliar", "triliun", "kuadriliun", "kuintiliun", 
              "sekstiliun", "septiliun", "oktiliun", "noniliun", "desiliun"]
    
    words = []
    scale_index = 0
    
    while number > 0:
        chunk = number % 1000
        if chunk != 0:
            if scale_index == 0:
                words.append(convert_less_than_thousand(chunk))
            else:
                if chunk == 1 and scale_index == 1:  # Special case for 1000
                    words.append("seribu")
                else:
                    chunk_text = convert_less_than_thousand(chunk)
                    words.append(f"{chunk_text} {scales[scale_index]}")
        
        number = number // 1000
        scale_index += 1
    
    return ' '.join(reversed(words))
