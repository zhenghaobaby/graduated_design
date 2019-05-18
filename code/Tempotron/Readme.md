**overview**

1. this is the code for SNN-tempotron algorithm. 

2. **tempotronClassify.m** is the main program. and **myNerualNetworkFunction.m**  is a function which use the LTP data. (you put a x, it will return you the y=function(x), x can be a vector)

3. if you want to change the learning rule, change the **line 69** and **line 111** to the rule you want.

   (for example ,you want to use the STDP function, when you get the right STDP, you can fit the curve and get a function, then use this function in the code)