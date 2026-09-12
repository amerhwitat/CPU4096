package com.amerhwitat.cpu4096.mobile
import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.widget.*
class MainActivity : Activity() { override fun onCreate(savedInstanceState: Bundle?) { super.onCreate(savedInstanceState); val root=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;gravity=Gravity.CENTER;setPadding(32,32,32,32)}; val title=TextView(this).apply{text="CPU4096 — Kotlin Mobile";textSize=24f;gravity=Gravity.CENTER}; val state=TextView(this).apply{text="Wide-register simulator\n4096/8192-bit research path\n128D state: ready";textSize=16f;gravity=Gravity.CENTER;setPadding(0,24,0,24)}; val action=Button(this).apply{text="Initialize CPU";setOnClickListener{state.text="CPU model: active\nRegister width: 4096-bit baseline\n128D state: active"}};root.addView(title);root.addView(state);root.addView(action);setContentView(root)} }
