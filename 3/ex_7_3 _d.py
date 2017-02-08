############################################
# Python 2.7
# Ex: 7.3
# PL-TRUE? algorithm
# Last modified on 8th Nov 1:34 AM
# Author Praneeta Mallela, pmallela@wpi.edu
# PLEASE INSTALL SYMPY LIBRARY BRFORE RUNNIN THE CODE
############################################
#  1 = True; 0 = False

import re
import numpy as np
from sympy import *
import time
import matplotlib.pyplot as plt

def getOpArgs(s):
	# orderDictionary = {1:'~', 2: '&', 3: '|', 4: '==>', 5: '<==', 6:'<=>'}
	args = [x for x in list(s) if x.isalpha()]
	ops = [x for x in list(s) if not (x.isalpha())]
	if not ops: return None, args
	# print ops, args
	return ops[0],args

def lookUp(s,m):
	if s in m:
		return m[s]
	else:
		return None

def getOp(expr_final):
	return [x for x in list(expr_final) if not (x.isalpha())]

def isSymbol(sym):
	return sym.isalpha()

def PL_TRUE(s,m):
	if s == True or s ==False: return s
	op,args  = getOpArgs(s)
	if isSymbol(s) is True:return lookUp(s,m)
	elif op == '~':
		if len(args) ==1:
			pl = PL_TRUE(args[0],m)
			return not pl
		else:
			pl = PL_TRUE(s.split('~',1)[1],m)
			return pl
	elif op == '|':
		plf = PL_TRUE(args[0],m) 
		plb= PL_TRUE(s.split('|',1)[1],m)
		return plf or plb
	else:
		return None

if __name__=='__main__':
	m = {'a':1, 'b':1, 'k': 0, 'l':0, 'v':1, 'f':1, 'c':0, 'h':0, 's':0,'r':1,
	'p':0, 'e':1, 'w':0, 'g':1,'f':0,'d':1,'q':1}
	# sentences = ['p',True,'~a|m',   'g&k<<b>>~c>>lw<=>g|e', 'a<=>b|e&(s>>f)>>(s&h)>>f', 'e&~h<<d|r&f|~h&~w'] 
	# sentences = ['r&e<=>c','g&k<<b>>~k&s|f|p','w<=>g|e>>q>>p','q<<r&k|b&~c>>l',]
	sentences = ['~a|z', '~a&z', 'a|z','a&z', 'z<=>z','z&~z', 'z|~z' ]
	time_axis = []
	len_sentences = []; 
	for expr in sentences:
		print "Sentence=", expr,
		if expr == True or expr == False:
			start_time = time.time()
			ans = PL_TRUE(expr,m)
			len_sentences.append(1)
			time_axis.append(time.time()-start_time)
		else:

			length_exp = 0
			start_time = time.time()
			expr_split = expr.split('<=>')
			expr = ' '
			for indx in range(len(expr_split)-1):
				expr = expr+expr_split[indx] + '>>'+ expr_split[indx+1][0]+'&'+expr_split[indx][len(expr_split[indx])-1]+'<<'
			expr_final = expr+expr_split[len(expr_split)-1]
			expr_x = to_cnf(expr_final)
			args_x = expr_x.args
			if '&' in getOp(expr_final): flag = 1
			else: flag = 0
			none_ans  = 0
			if flag:
				ans = 1
				for arg in args_x:
					sent = str(arg).replace('Not(','~').replace(')','').replace(',','|').replace('Or(','').replace(' ','')
					trueFalse = PL_TRUE(sent,m)
					if trueFalse!= None :
						ans = ans & trueFalse
					elif trueFalse == None:
						none_ans = 1
						ans = ans
					else:
						ans = None
						break
					length_exp = length_exp+1
				if none_ans and ans:
					ans = None
				elif none_ans and not ans:
					ans = 0
				else:
					ans = ans
			else:
				ans = 0
				for arg in args_x:
					sent = str(arg).replace('Not(','~').replace(')','').replace('And(','').replace(',','&').replace(' ','')
					trueFalse = PL_TRUE(sent,m)
					if trueFalse!= None:	
						ans = ans | trueFalse
					elif trueFalse == None:
						none_ans = 1
						ans = ans
					else: 
						ans = None
						break

				if none_ans and ans:
					ans = 1
				elif none_ans and not ans:
					ans = None
				else:
					ans = ans
					length_exp = length_exp+1
			time_axis.append(time.time() - start_time)
			len_sentences.append(length_exp)
		print "Value = ",ans
	plt.plot(time_axis, len_sentences)
	plt.xlabel('time')
	plt.ylabel('length of sentence')
	plt.show()