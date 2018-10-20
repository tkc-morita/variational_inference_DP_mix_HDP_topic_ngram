# coding: utf-8

def encode_data(data, encoder, add_end_symbol = True):
	if add_end_symbol:
		return [tuple([encoder[symbol] for symbol in string.split(',')+['END'] if symbol]) for string in data]
	else:
		return [tuple([encoder[symbol] for symbol in string.split(',') if symbol]) for string in data]

def decode_data(data, decoder, end_code = None):
	if end_code is None:
		return [tuple([decoder[code] for code in string]) for string in data]
	else:
		return [tuple([decoder[code] for code in string+(end_code,)]) for string in data]


def df2coder(df_code):
	df_encoder = df_code.set_index('symbol')
	encoder = df_encoder.code.to_dict()

	df_decoder = df_code.set_index('code')
	decoder = df_decoder.symbol.to_dict()

	return encoder, decoder