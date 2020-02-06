import { Component, OnInit } from '@angular/core';

export interface PropertyElement {
  name: string;
  value: string;
}

export interface GrowthSectorElement {
  sector: string;
  ir: string;
  weight: string;
}

const DATA1: PropertyElement[] = [
  {name: 'FCR', value: '3.1'},
  {name: '비육 사료량 비중', value: '82%'},
  {name: '도체율', value: '74.7%'}
];

const DATA2: GrowthSectorElement[] = [
  {sector: '자돈', ir: '0.650%', weight: '7.0%'},
  {sector: '육성돈', ir: '0.379%', weight: '25.0%'},
  {sector: '비육돈', ir: '0.244%', weight: '50.0%'},
  {sector: '임신돈', ir: '0.129%', weight: '12.0%'},
  {sector: '포유돈', ir: '0.401%', weight: '6.0%'}
]

@Component({
  selector: 'app-demo',
  templateUrl: './demo.component.html',
  styleUrls: ['./demo.component.css']
})
export class DemoComponent implements OnInit {
  displayedColumns1: string[] = ['name', 'value'];
  displayedColumns2: string[] = ['sector', 'ir', 'weight'];
  dataSource1 = DATA1;
  dataSource2 = DATA2;
  constructor() { }

  ngOnInit() {
  }

}
