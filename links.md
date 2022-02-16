---
layout: page
title: Social Links
subtitle: My works so far
---

## Links
Here are some useful links that I use actively

### Notion

{: .box-note}
**Note:** Notion is where I store all my study materials and interests. Please go ahead and check it out!

[<img src="https://cdn.worldvectorlogo.com/logos/notion-logo-1.svg" alt="drawing" style="width:200px;"/>](https://sunbinmun.notion.site/Sun-Bin-MUN-Getting-Started-1c4a5242fd3d4a2ca157510f5318ae7d){: .mx-auto.d-block :}

### Github
{: .box-note}
**Note:** I am still trying to learn more about the Github. It is still in progress!

[<img src="https://logos-world.net/wp-content/uploads/2020/11/GitHub-Emblem.png" alt="drawing" style="width:200px;"/>](https://github.com/msb1002){: .mx-auto.d-block :}


### Request for CV
If you want to request for the CV, please fill in the following information :)

<input type="text" id="name" name="name"/>

<label for="name">Name:</label> 
<input type="text" id="name" name="name" size="50"/>

<label for="city">City:</label>
<select id="city" name="city">
  <option value="BOS">BOS</option>
  <option value="SFO">SFO</option>
  <option value="NYC" selected="selected">NYC</option>
</select>

<!---Reference:https://codepen.io/meodai/pen/rNedxBa--->

<article class="l-design-widht">
  <h1>Mini CSS-vars Design System</h1>

  <p>Most of the projects I work on have about <mark>3</mark> important colors: Main- , Accent-  and Background-Color. Naturally tons of other colors are used in a typical Project, but they are mostly used within specific components.</p>
  
  <p>I find it useful to set those 3 colors as vars on the root and change them in specific contexts. It turns out that the complexity of the components I build is dramatically cut down by this. And also gives me a lot of control over the cascade.</p>
  
  <div class="card">
    <h2><svg class="icon" aria-hidden="true">
      <use xlink:href="#icon-coffee" href="#icon-coffee" /></svg>Regular</h2>
    <label class="input">
      <input class="input__field" type="text" placeholder=" " />
      <span class="input__label">Some Fancy Label</span>
    </label>
    <div class="button-group">
      <button>Send</button>
      <button type="reset">Reset</button>
    </div>
  </div>
  <div class="card card--inverted">
    <h2> <svg class="icon" aria-hidden="true">
      <use xlink:href="#icon-coffee" href="#icon-coffee" />
    </svg>Inverted</h2>
    <label class="input">
      <input class="input__field" type="text" placeholder=" " value="Valuable value" />
      <span class="input__label">Some Fancy Label</span>
    </label>
   
    <div class="button-group">
      <button>Send</button>
      <button type="reset">Reset</button>
    </div>
  </div>
    <div class="card card--accent">
    <h2><svg class="icon" aria-hidden="true">
      <use xlink:href="#icon-coffee" href="#icon-coffee" />
    </svg>Accent</h2>
    <label class="input">
      <input class="input__field" type="text" placeholder=" " />
      <span class="input__label">Some Fancy Label</span>
    </label>
    
    <div class="button-group">
      <button>Send</button>
      <button type="reset">Reset</button>
    </div>
  </div>
  
    <div class="card card--inverted">
    <h2>Colors</h2>
    <p>Play around with the colors</p>
    <input type="color" data-color="light" value="#ffffff" />
    <input type="color" data-color="dark" value="#212121" />
    <input type="color" data-color="signal" value="#fab700" />
  </div>
</article>

<svg xmlns="http://www.w3.org/2000/svg" class="hidden">
  <symbol id="icon-coffee" viewBox="0 0 20 20">
    <title>icon-coffee</title>
    <path fill="currentColor" d="M15,17H14V9h3a3,3,0,0,1,3,3h0A5,5,0,0,1,15,17Zm1-6v3.83A3,3,0,0,0,18,12a1,1,0,0,0-1-1Z"/>
    <rect fill="currentColor" x="1" y="7" width="15" height="12" rx="3" ry="3"/>
    <path fill="var(--color-accent)" d="M7.07,5.42a5.45,5.45,0,0,1,0-4.85,1,1,0,0,1,1.79.89,3.44,3.44,0,0,0,0,3.06,1,1,0,0,1-1.79.89Z"/>
    <path fill="var(--color-accent)" d="M3.07,5.42a5.45,5.45,0,0,1,0-4.85,1,1,0,0,1,1.79.89,3.44,3.44,0,0,0,0,3.06,1,1,0,1,1-1.79.89Z"/>
    <path fill="var(--color-accent)" d="M11.07,5.42a5.45,5.45,0,0,1,0-4.85,1,1,0,0,1,1.79.89,3.44,3.44,0,0,0,0,3.06,1,1,0,1,1-1.79.89Z"/>
  </symbol>
</svg>

<input class="c-checkbox" type="checkbox" id="checkbox">
<div class="c-formContainer">
  <form class="c-form" action="">
    <input class="c-form__input" placeholder="E-mail" type="email" pattern="[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{1,63}$" required>
    <label class="c-form__buttonLabel" for="checkbox">
      <button class="c-form__button" type="button">Send</button>
    </label>
    <label class="c-form__toggle" for="checkbox" data-title="Notify me"></label>
  </form>
</div>

<label for="inp" class="inp">
  <input type="password" id="inp" placeholder="Password" pattern=".{6,}" required>
  <svg width="280px" height="18px" viewBox="0 0 280 18" class="border">
    <path d="M0,12 L223.166144,12 C217.241379,12 217.899687,12 225.141066,12 C236.003135,12 241.9279,12 249.827586,12 C257.727273,12 264.639498,12 274.514107,12 C281.097179,12 281.755486,12 276.489028,12"></path>
  </svg>
  <svg width="14px" height="12px" viewBox="0 0 14 12" class="check">
    <path d="M1 7 5.5 11 L13 1"></path>
  </svg>
</label>
</form>

<div class="page">
  <label class="field field_v1">
    <input class="field__input" placeholder="e.g. Stanislav">
    <span class="field__label-wrap">
      <span class="field__label">First name</span>
    </span>
  </label>
  <label class="field field_v2">
    <input class="field__input" placeholder="e.g. Melnikov">
    <span class="field__label-wrap">
      <span class="field__label">Last name</span>
    </span>
  </label>    
  <label class="field field_v3">
    <input class="field__input" placeholder="e.g. melnik909@ya.ru">
    <span class="field__label-wrap">
      <span class="field__label">E-mail</span>
    </span>
  </label>
</div>
<div class="linktr">
  <a href="https://linktr.ee/melnik909" target="_blank" class="linktr__goal r-link">Get more things from me 💪💪💪</a>
</div>
